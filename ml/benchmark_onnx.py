"""Repeatable CPU latency measurement for full ONNX outputs (including heatmaps)."""
import argparse, json, os, platform, statistics, subprocess, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def cpu_name():
    if os.name=='nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key,'ProcessorNameString')[0]
        except Exception: pass
    return platform.processor() or platform.machine()

def run_model(path,x,warmup,repeats,threads):
    opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    for _ in range(warmup): sess.run(None,{'images':x})
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); out=sess.run(None,{'images':x}); times.append((time.perf_counter()-t)*1000)
    return {'file':str(path),'input_shape':list(x.shape),'output_shapes':[list(v.shape) for v in out],
            'warmup_runs':warmup,'measured_runs':repeats,'threads_intra_op':threads,'threads_inter_op':1,
            'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),'std_ms':statistics.pstdev(times),
            'p95_ms':float(np.percentile(times,95)),'min_ms':min(times),'max_ms':max(times)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',default=['outputs/ObjectLab/models/baseline.onnx','outputs/ObjectLab/models/sota.onnx']); ap.add_argument('--out',default='outputs/ObjectLab/results/latency_benchmark.json'); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--runs',type=int,default=50); ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args(); x=np.random.default_rng(2).random((1,3,224,224),dtype=np.float32)
    report={'scope':'ONNX graph only: RGB float tensor in, probabilities and all class heatmaps out; excludes image decode, resize/pad, and application overhead. Same N=1 input for both models.','os':platform.platform(),'processor':cpu_name(),'python':platform.python_version(),'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','models':[run_model(Path(p),x,a.warmup,a.runs,a.threads) for p in a.models]}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__': main()
