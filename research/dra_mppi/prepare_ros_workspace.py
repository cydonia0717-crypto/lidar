"""Build-only copies and explicit compatibility patches; leave upstream intact."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    packages = {'asr_rapidxml': 'src/asr_rapidxml', 'ros_tools': 'src/ros_tools',
                'pedsim_original': 'src/pedsim_original', 'mpc_planner_msgs': 'src/mpc_planner/mpc_planner_msgs',
                'pedestrian_simulator': 'src/pedestrian_simulator'}
    for name, relative in packages.items():
        shutil.copytree(a.upstream/relative, a.workspace/'src'/name)
    shutil.copytree(a.harness/'ros_bridge', a.workspace/'src/dra_repro_bridge')
    patches = []
    def patch(path, old, new, reason):
        before = path.read_text()
        if before.count(old) != 1:
            raise AssertionError('Patch anchor is not unique: '+str(path))
        after = before.replace(old, new, 1)
        path.write_text(after)
        patches.append({'file': str(path.relative_to(a.workspace)), 'reason': reason,
                        'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(after.encode()).hexdigest()})
        with (a.output/'build_compatibility.patch').open('a') as f:
            f.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(path), tofile=str(path)))
    msg = a.workspace/'src/mpc_planner_msgs'
    patch(msg/'CMakeLists.txt', 'find_package(catkin REQUIRED COMPONENTS',
          'find_package(catkin REQUIRED COMPONENTS\n  message_generation', 'Declare message generation explicitly for the isolated build')
    patch(msg/'package.xml', '<buildtool_depend>catkin</buildtool_depend>',
          '<buildtool_depend>catkin</buildtool_depend>\n  <build_depend>message_generation</build_depend>\n  <exec_depend>message_runtime</exec_depend>', 'Message build/runtime dependency metadata')
    (a.output/'build_compatibility.json').write_text(json.dumps(patches, indent=2))


if __name__ == '__main__':
    main()
