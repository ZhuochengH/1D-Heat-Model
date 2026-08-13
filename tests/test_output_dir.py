"""
STEP 10B: 规范输出目录解析测试。

保护的行为:
1. 默认输出目录 = <repo root>/calibration_output/ (基于 __file__, 与 cwd 无关)
2. 改变当前工作目录 (cwd) 不改变默认输出目录
3. 显式 --output-dir 覆盖默认
   (绝对路径原样使用; 相对路径保持相对 cwd 的历史语义)
"""

from pathlib import Path

from peltier_surface_calibration_v2 import PROJECT_ROOT, resolve_output_dir


def test_project_root_is_repo_root():
    # 测试文件位于 <repo root>/tests/, 其父的父即仓库根
    expected = Path(__file__).resolve().parents[1]
    assert PROJECT_ROOT == expected
    # 源文件确实位于 <repo root>/Surface_calibration/ 下
    assert (
        PROJECT_ROOT
        / "Surface_calibration"
        / "peltier_surface_calibration_v2.py"
    ).is_file()


def test_default_output_dir_is_repo_root_calibration_output():
    assert resolve_output_dir(None) == PROJECT_ROOT / "calibration_output"


def test_default_independent_of_cwd(monkeypatch, tmp_path):
    default_from_repo = resolve_output_dir(None)
    # 改变 cwd 到任意临时目录
    monkeypatch.chdir(tmp_path)
    default_from_elsewhere = resolve_output_dir(None)
    assert default_from_elsewhere == default_from_repo
    assert default_from_elsewhere == PROJECT_ROOT / "calibration_output"


def test_explicit_absolute_output_dir_overrides_default(tmp_path):
    custom = tmp_path / "custom_output"
    assert resolve_output_dir(str(custom)) == custom
    assert resolve_output_dir(str(custom)) != resolve_output_dir(None)


def test_explicit_relative_output_dir_resolves_against_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rel = resolve_output_dir("my_output")
    # 保持历史语义: 显式相对路径相对当前工作目录解析
    assert rel == Path("my_output")
    assert rel != resolve_output_dir(None)
