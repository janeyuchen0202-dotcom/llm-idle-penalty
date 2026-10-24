"""組出 mac_idle_test.sh：外層啟動檔（wrapper_template.sh）＋主程式（mac_idle_test_template.py 填入提示詞）。換行一律 LF（Mac 的 bash 讀 CRLF 會出錯）"""
import ast, os, sys
b = os.path.dirname(os.path.abspath(__file__))
py = open(os.path.join(b, "mac_idle_test_template.py"), encoding="utf-8").read()
a = '        args = ["-m", MODEL, "-c", "4096", "--parallel", "1", "-t", "4", "--cache-ram", "0", "-ngl", "all"]\n'
assert py.count(a) == 1
py = py.replace(a, a + '        args += arg_value("--server-args", "").split()                  # 只給作者試跑用\n')
p64 = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(b, "prompts_b64.txt")))
py = py.replace("__MEDIUM_B64__", p64["medium.txt"]).replace("__SHORT_B64__", p64["short.txt"])
ast.parse(py)
assert "PY_EOF" not in py and "\r" not in py
open(os.path.join(b, "mac_idle_test.py"), "w", encoding="utf-8", newline="\n").write(py)
sh = open(os.path.join(b, "wrapper_template.sh"), encoding="utf-8").read().replace("\r\n", "\n")
assert sh.count("__PYTHON_CLIENT__") == 1
sh = sh.replace("__PYTHON_CLIENT__", py.rstrip("\n"))
open(os.path.join(b, "mac_idle_test.sh"), "wb").write(sh.encode("utf-8"))
print("mac_idle_test.sh：", len(sh.encode("utf-8")), "bytes；CRLF" if "\r" in sh else "bytes；LF")
