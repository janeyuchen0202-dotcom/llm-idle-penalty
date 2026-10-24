"""實驗期間暫時阻止系統休眠（不改系統設定，程式結束就恢復）"""
import ctypes, sys, time
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 1800)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
