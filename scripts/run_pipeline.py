import argparse
import time
import platform
import torch
import psutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config,output_dir,save_json
from src.data.protocol import create_protocol
from src.data.corruptions import create_benchmark
from src.diffusion.train import train
from src.pipeline import build_memory,fit_models,calibrate,evaluate,ablate

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/smoke.yaml")
    parser.add_argument("--resume")
    args=parser.parse_args();config=load_config(args.config);out=output_dir(config)
    save_json(out/"run_config.json",config)
    started=time.perf_counter();stages={}
    for name,function in [("split",create_protocol),("train",lambda:train(config,args.resume)),
                          ("memory",lambda:build_memory(config)),("normal_models",lambda:fit_models(config)),
                          ("calibration",lambda:calibrate(config)),("test_benchmark",lambda:create_benchmark(config)),
                          ("evaluation",lambda:evaluate(config)),("ablations",lambda:ablate(config))]:
        before=time.perf_counter();print("STAGE",name,flush=True);function();stages[name]=time.perf_counter()-before
        save_json(out/"stage_timings.json",stages)
    save_json(out/"run_summary.json",{"seconds":time.perf_counter()-started,"stages":stages,
      "torch":torch.__version__,"cuda":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "cpu":platform.processor(),"ram_gb":psutil.virtual_memory().total/2**30,"config":config,"provisional":config["provisional"]})
