#!/usr/bin/env python3
"""
Script to automatically run all YAML configs (except config.yaml) with Hydra multirun.
Each config will output to ../../results/{config_name}/
"""

import os
import glob
import subprocess
import sys
import argparse
import actdyn


def find_yaml_configs(conf_dir="conf"):
    """Find all .yaml files in conf directory except config.yaml"""
    yaml_pattern = os.path.join(conf_dir, "*.yaml")

    yaml_files = glob.glob(yaml_pattern)
    config_names = []

    for yaml_file in yaml_files:
        basename = os.path.basename(yaml_file)
        config_name = os.path.splitext(basename)[0]

        # Skip config.yaml as it's the base config
        if config_name != "config":
            config_names.append(config_name)

    return sorted(config_names)


def run_hydra_multirun(conf_dir, config_name, script_path, results_base_dir):
    """Run a single config with Hydra multirun"""

    # Create the output directory path: results_base_dir/{config_name}
    output_dir = os.path.join(results_base_dir, config_name)

    print(f"\n{'='*60}")
    print(f"Running config: {config_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Construct the hydra command
    cmd = [
        "python",
        f"{script_path}",
        f"--config-path={conf_dir}",
        f"--config-name={config_name}",
        "--multirun",
        f"hydra.sweep.dir={output_dir}",
        f"hydra.run.dir={output_dir}/single_run",
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the command
        subprocess.run(cmd, check=True, capture_output=False, text=True)  # Show output in real-time
        print(f"✅ Successfully completed: {config_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {config_name}: {e}")
        print(f"Return code: {e.returncode}")
        return False
    except (OSError, FileNotFoundError) as e:
        print(f"❌ Command execution error for {config_name}: {e}")
        return False


def main():
    """Main function to run all configs"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run all YAML configs (except config.yaml) with Hydra multirun"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Directory containing the experiment script and conf/ folder. If not provided, uses the directory of this script.",
    )
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/run_experiment.py",
        help="Script to run with Hydra multirun. Default is 'run_experiment.py'.",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Config file to use for Hydra multirun.",
    )
    args = parser.parse_args()

    # Setup paths
    if args.path:
        experiment_dir = os.path.abspath(args.path)
        if not os.path.exists(experiment_dir):
            print(f"Error: Provided experiment directory does not exist: {experiment_dir}")
            sys.exit(1)
    else:
        print("Using default experiment directory")
        experiment_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the folder name of the config directory
    experiment_name = os.path.basename(experiment_dir)

    # Find actdyn installation and construct results directory dynamically
    actdyn_path = os.path.dirname(actdyn.__file__)
    project_root = os.path.dirname(actdyn_path)
    results_base_dir = os.path.join(project_root, "results", experiment_name)
    results_abs_dir = os.path.abspath(results_base_dir)
    script_path = os.path.abspath(args.script)

    # Check if conf directory exists
    conf_dir = os.path.join(experiment_dir, "conf")
    if not os.path.exists(conf_dir):
        print(f"Error: conf/ directory not found in: {experiment_dir}")
        sys.exit(1)

    if args.config:
        config_names = [args.config]
    else:
        config_names = find_yaml_configs(conf_dir)

    if not config_names:
        print("No config files found (other than config.yaml)")
        return

    print(f"\nFound {len(config_names)} config file(s):")
    for config_name in config_names:
        print(f"  - {config_name}.yaml")

    # Ask for confirmation
    response = input(f"\nRun all {len(config_names)} configs? [y/N]: ").strip().lower()
    if response not in ["y", "yes"]:
        print("Cancelled.")
        return

    # Create results base directory
    os.makedirs(results_abs_dir, exist_ok=True)

    # Track results
    successful = []
    failed = []

    # Change to script directory to run commands
    original_cwd = os.getcwd()
    os.chdir(experiment_dir)

    try:
        # Run each config
        for i, config_name in enumerate(config_names, 1):
            print(f"\n[{i}/{len(config_names)}] Processing {config_name}...")

            if run_hydra_multirun(conf_dir, config_name, script_path, results_abs_dir):
                successful.append(config_name)
            else:
                failed.append(config_name)
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total configs: {len(config_names)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n✅ Successful configs:")
        for config in successful:
            print(f"  - {config}")

    if failed:
        print("\n❌ Failed configs:")
        for config in failed:
            print(f"  - {config}")

    print(f"\nResults are saved in: {results_abs_dir}")


if __name__ == "__main__":
    main()
