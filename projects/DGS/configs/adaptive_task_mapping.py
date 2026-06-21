import argparse
import sys
import os

# Add repo root so custom_imports (projects.DGS.dgs) resolves correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from projects.DGS.dgs.domain_predictor.builder import AdaptiveDomainPredictor
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Update task_id_mapping before training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--task_id', type=int, required=True, help='Current task id')
    parser.add_argument('--seen_tasks', type=str, required=True, help='Comma separated list of seen tasks')
    parser.add_argument('--num_tasks', type=int, required=True)
    parser.add_argument('--work_dirs', type=str, required=False, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seen_tasks = [x for x in args.seen_tasks.split(',') if x]
    cfg = Config.fromfile(args.config)
    domain_predictor_cfg = cfg.model.domain_predictor_cfg
    predictor = AdaptiveDomainPredictor(domain_predictor_cfg, 
                                        num_tasks=args.num_tasks, 
                                        moe_modules=[], 
                                        task_id=args.task_id, 
                                        seen_tasks=seen_tasks,
                                        work_dirs=args.work_dirs)