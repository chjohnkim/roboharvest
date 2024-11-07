import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-m', '--mode', type=str)
def main(session_dir, mode):
    script_dir = pathlib.Path(__file__).parent.joinpath('data_processing')
    for session in session_dir:
        session = pathlib.Path(session)
        demos_dir = session.joinpath('demos')
        
        # run gripper range calibration
        if mode == 'gripper':
            script_path = script_dir.joinpath('calibrate_gripper_range.py')
        elif mode.startswith('pruner'):
            script_path = script_dir.joinpath('calibrate_pruner_range.py')
        assert script_path.is_file()
        
        for gripper_dir in demos_dir.glob("gripper_calibration*"):
            gripper_range_path = gripper_dir.joinpath('gripper_range.json')
            tag_path = gripper_dir.joinpath('tag_detection.pkl')
            assert tag_path.is_file()
            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path)
            ]
            subprocess.run(cmd)

            
# %%
if __name__ == "__main__":
    main()
