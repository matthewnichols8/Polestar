import os
import logging
import multiprocessing as mp
import traceback
from queue import Empty
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig
import socket
from typing import Any, Optional, List, Tuple
import random
import time
from contextlib import closing


from .parallel_strategy import ThresholdParallelDecoder
from .utils import KVCacheFactory, BlockIteratorFactory
from .generate_uniform import BlockWiseDiffusionLLM
from ..model.modeling_llada import LLaDAModelLM

logger = logging.getLogger(__name__)

def is_port_available(port: int) -> bool:
        """
        Check if a single port is available.

        :param port: The port number to check.
        :return: Whether the port is available.
        """
        sock_type = socket.SOCK_STREAM
        try:
            with closing(socket.socket(socket.AF_INET, sock_type)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False

def find_continuous_ports(
        num_ports: int,
        start_port: int = 30000,
        end_port: int = 65535,
        protocol: str = 'tcp',
        max_attempts: int = 300,
        retry_delay: float = 0.1
    ) -> Optional[Tuple[int, List[int]]]:
    """
    Find a sequence of consecutive free ports within a specified range.

    :param num_ports: Number of consecutive ports required.
    :param start_port: Starting port number (default: 1024).
    :param end_port: Ending port number (default: 65535).
    :param protocol: Protocol type ('tcp' only).
    :param max_attempts: Maximum number of retry attempts.
    :param retry_delay: Delay between retries (in seconds).
    :return: A tuple of (start_port, list_of_ports) if found, otherwise raise ValueError.
    """
    # Validate input parameters
    if num_ports <= 0:
        raise ValueError("Number of ports must be greater than 0")
    if start_port < 0 or end_port > 65535:
        raise ValueError("Port range must be between 0 and 65535")
    if start_port > end_port:
        raise ValueError("Start port cannot be greater than end port")
    if (end_port - start_port + 1) < num_ports:
        raise ValueError(f"Port range ({start_port}-{end_port}) is too small to accommodate {num_ports} consecutive ports")

    # calculate the max start port
    max_start = end_port - num_ports + 1

    for attempt in range(max_attempts):
        # Seeding with a combination of process ID and high-resolution timestamp
        pid = os.getpid()
        seed = pid + int(time.time() * 1_000_000)
        random.seed(seed)

        current = random.randint(start_port, max_start-1)
        while current <= max_start:
            # check current port
            if not is_port_available(current):
                current += 1
                continue

            # check following n+1 port
            all_available = True
            for offset in range(1, num_ports):
                port_to_check = current + offset
                if not is_port_available(port_to_check):
                    all_available = False
                    current = port_to_check + 1
                    break

            # find available port and then lock these port
            if all_available:
                # check port again
                final_check = True
                for port in range(current, current + num_ports):
                    if not is_port_available(port):
                        final_check = False
                        current = port + 1
                        break

                if final_check:
                    ports = list(range(current, current + num_ports))
                    return current, ports

        # No port found in this attempt; sleep and retry.
        time.sleep(retry_delay)

    # No port found in all attempts; raise valueerror
    raise ValueError(f"No available ports found after {max_attempts} attempts. Range: {start_port}-{end_port}, Number of ports: {num_ports}")


class SamplingParams:
    """Sampling settings for the LLaDA threshold and Polestar decoders."""

    def __init__(
        self,
        *,
        parallel_decoding="threshold",
        cache="prefix",
        threshold=0.9,
        temperature=0.0,
        early_stop=True,
        cont_weight=0.0,
        use_bd=False,
        enable_torch_compile=True,
        mask_id=126336,
        eos_id=126081,
        current_commit_lower_bound=0.7,
        polestar_steps=None,
        polestar_seed=42,
        polestar_profile_timing=False,
        polestar_refresh_after_tokens=3,
        polestar_refresh_layer_count=None,
    ):
        self.parallel_decoding = parallel_decoding
        self.cache = cache
        self.threshold = threshold
        self.temperature = temperature
        self.early_stop = early_stop
        self.cont_weight = cont_weight
        self.use_bd = use_bd
        self.enable_torch_compile = enable_torch_compile
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.current_commit_lower_bound = current_commit_lower_bound
        self.polestar_steps = polestar_steps
        self.polestar_seed = polestar_seed
        self.polestar_profile_timing = polestar_profile_timing
        self.polestar_refresh_after_tokens = polestar_refresh_after_tokens
        self.polestar_refresh_layer_count = polestar_refresh_layer_count


def init_generator(model, sample_params, backend="vllm", max_length=4096):
    """Select the tested Polestar path or dInfer threshold/prefix baseline."""
    if sample_params.parallel_decoding == "polestar":
        if sample_params.cache != "polestar":
            raise ValueError("Polestar Cache and Commit require cache='polestar'")
        if sample_params.temperature != 0.0:
            raise ValueError("Polestar dInfer decoding requires temperature=0")
        if sample_params.use_bd or sample_params.cont_weight != 0:
            raise ValueError("Polestar dInfer decoding does not use BD or continuous weighting")
        if not sample_params.early_stop or not sample_params.enable_torch_compile:
            raise ValueError("Polestar dInfer decoding requires early_stop and model compilation")
        from polestar.integrations.dinfer_commit import make_polestar_drift_generator

        return make_polestar_drift_generator(
            model,
            threshold=sample_params.threshold,
            lower_bound=sample_params.current_commit_lower_bound,
            mask_id=sample_params.mask_id,
            eos_id=sample_params.eos_id,
            refresh_after_tokens=sample_params.polestar_refresh_after_tokens,
            refresh_layer_count=sample_params.polestar_refresh_layer_count,
            request_seed=sample_params.polestar_seed,
            steps=sample_params.polestar_steps,
        )

    if (
        sample_params.parallel_decoding != "threshold"
        or sample_params.cache != "prefix"
        or sample_params.use_bd
        or sample_params.cont_weight != 0
    ):
        raise ValueError("The bundled dInfer baseline requires threshold decoding and a prefix cache")

    decoder = ThresholdParallelDecoder(
        temperature=sample_params.temperature,
        threshold=sample_params.threshold,
        mask_id=sample_params.mask_id,
        eos_id=sample_params.eos_id,
    )
    return BlockWiseDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True),
        cache_factory=KVCacheFactory("prefix", backend=backend, max_length=max_length),
        early_stop=sample_params.early_stop,
    )


def generate(dllm, device, req_q, res_q, profile_timing=False):
    while True:
        data = req_q.get()
        if isinstance(data, str):
            assert data == 'stop'
            break
        else:
            input_ids, gen_len, block_len = data
        input_ids = input_ids.to(device)
        if profile_timing:
            torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
        out = dllm.generate(input_ids, gen_length=gen_len, block_length=block_len)
        if profile_timing:
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - decode_start
        num_forwards = dllm.num_forwards
        if res_q is not None:
            res_q.put((out, num_forwards, decode_seconds) if profile_timing else (out, num_forwards))

def server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port, error_q=None):
    try:
        _server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port)
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[SERVER PROCESS {rank} ERROR]: {error_msg}")
        if error_q is not None:
            try:
                error_q.put((e, error_msg))
            except:
                logger.error(f"[ERROR_Q exception in {rank}].")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

def _server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    is_polestar = sample_params.parallel_decoding == 'polestar'
    if is_polestar and world_size != 1:
        raise ValueError(
            "The Polestar dInfer cache supports one GPU per worker"
        )
    device = torch.device(gpu_id)
    if is_polestar:
        seed = int(sample_params.polestar_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    if world_size > 1:
        model.tensor_parallel(rank, world_size)
    if sample_params.enable_torch_compile or is_polestar:
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
    model = model.to(device)

    dllm = init_generator(model, sample_params)
    generate(dllm, model.device, req_q=q, res_q=res_q, profile_timing=sample_params.polestar_profile_timing)

    dist.destroy_process_group()

class ServerGroup:
    def __init__(self):
        self.procs = []
        self.req_qs = []
        self.res_q = None
        self.error_qs = []

    def add_request(self, req):
        assert len(self.req_qs) != 0 and len(self.req_qs) == len(self.procs)
        for q in self.req_qs:
            q.put(req)

    def get_response(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for i, eq in enumerate(self.error_qs):
                try:
                    error, error_msg = eq.get_nowait()
                except Empty:
                    continue
                raise RuntimeError(f"Worker {i} failed:\n{error_msg}") from error
            if any(p.exitcode not in (None, 0) for p in self.procs):
                raise RuntimeError("Inference worker exited before returning a response")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"Inference did not return within {timeout} seconds")
            try:
                return self.res_q.get(timeout=min(0.25, remaining) if remaining is not None else 0.25)
            except Empty:
                continue

    def start_server(self, model_path, sample_params, server_port, gpus):
        ctx = mp.get_context('spawn')
        assert len(self.procs) == 0, 'The server is already running.'
        procs = []
        req_qs = []
        error_qs = []
        for i, gpu in enumerate(gpus):
            if i == 0:
                res_q = ctx.Queue()
                self.res_q = res_q
            else:
                res_q = None
            q = ctx.Queue()
            req_qs.append(q)
            error_q = ctx.Queue()
            error_qs.append(error_q)
            p = ctx.Process(
                target=server_process,
                args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port, error_q),
            )
            p.daemon = True
            procs.append(p)
            p.start()
        self.procs = procs
        self.req_qs = req_qs
        self.error_qs = error_qs

    def is_running(self):
        return len(self.procs) != 0

    def stop_running(self):
        for q in self.req_qs:
            q.put('stop')
            q.close()
            q.join_thread()
        for p in self.procs:
            p.join(timeout=20)
            if p.is_alive():
                logger.warning("Inference worker did not stop within 20 seconds; terminating it")
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
        for q in [*self.error_qs, self.res_q]:
            if q is not None:
                q.close()
                q.join_thread()
        self.procs = []
        self.req_qs = []
        self.error_qs = []
        self.res_q = None

class ServerHandle:
    def __init__(self):
        self.groups = []
        self.need_response = 0

    def add_requests(self, reqs):
        prompts, gen_length, block_length = reqs
        self.need_response = 0
        # assert len(self.groups) == prompts.shape[0], 'We cannot only use DP to support batch size > 1.'
        if len(self.groups) == prompts.shape[0]:
            for i, prompt in enumerate(prompts):
                self.groups[i].add_request((prompt.unsqueeze(0), gen_length, block_length))
            self.need_response = len(prompts)
        else:
            partial_data = torch.chunk(prompts, len(self.groups), dim=0)
            for i in range(len(partial_data)):
                self.groups[i].add_request((partial_data[i], gen_length, block_length))
            self.need_response = len(partial_data)


    def get_responses(self, timeout=None):
        res = []
        for i in range(self.need_response):
            group = self.groups[i]
            res.append(group.get_response(timeout=timeout))
        return res

    def start_server(self, model_path, sample_params, server_port, num_gpus, dp_size, tpep_size):
        gpu = 0
        assert num_gpus >= dp_size * tpep_size
        for i in range(dp_size):
            self.groups.append(ServerGroup())
            gpus = [gpu + i for i in range(tpep_size)]
            logger.info(f'start server group on GPU {gpus}, server port: {server_port[i]}')
            self.groups[-1].start_server(model_path, sample_params, server_port[i], gpus)
            gpu = gpus[-1] + 1


    def is_running(self):
        return len(self.groups) != 0

    def stop_running(self):
        for group in self.groups:
            group.stop_running()
        self.groups = []

handle = ServerHandle()

class DiffusionLLMServing:
    """Serve LLaDA requests in dInfer worker processes.

    Parameters
    ----------
    model : str
        The model path
    sample_params : SamplingParams
        The parameters used in sampling.
    server_port: int
        The port for communication between the background process.
    num_gpus : int
        The number of GPUs used for parallel computation.
    max_retries : int
        The maximum number of retry attempts per DP group when searching for free ports.
    start_port : int
        The starting port number (inclusive) for the port search range.
    end_port : int
        The ending port number (inclusive) for the port search range.
    """
    def __init__(self, model, model_type='llada', sample_params=None, server_port=None, num_gpus=None, dp_size=None, tpep_size=None, backend='torch', timeout = None,
                max_retries=3, start_port=30000, end_port=60000
                ):
        if model_type != "llada" or backend != "torch":
            raise ValueError("The bundled dInfer worker supports LLaDA with backend='torch'")
        if sample_params is None:
            sample_params = SamplingParams()
        self.sample_params = sample_params
        if num_gpus is None:
            num_gpus = 1
        if dp_size is None:
            dp_size = 1
        if tpep_size is None:
            tpep_size = 1
        if tpep_size != 1:
            raise ValueError("The bundled dInfer worker supports one GPU per worker")
        if dp_size < 1 or dp_size > num_gpus:
            raise ValueError("dp_size must be between 1 and num_gpus")
        if server_port == None:
            # find a free port for each group
            server_port = self.port_selection(dp_size, start_port, end_port, max_retries=max_retries)
        else:
            # Keep compatibility with the existing implementation: assign ports starting from the current port + 1.
            server_port = [server_port+i for i in range(dp_size)]
        logger.info(f"find server port:{server_port}")

        if not handle.is_running():
            handle.start_server(model, sample_params, server_port, num_gpus, dp_size, tpep_size)
        self.num_forwards = 0
        self.timeout = timeout

    def generate(self, prompts, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompts: Torch.Tensor
            A tensor of shape (b, L) that contains the input prompts.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (b, L') that contains the prompt tokens and the generated tokens.
            The generation results of different lengths are padded with EOS.
        '''
        prompts = prompts.cpu()
        handle.add_requests((prompts, gen_length, block_length))
        rets = handle.get_responses(timeout=self.timeout)
        max_len = max(item[0].shape[1] for item in rets)
        total_batch_size = sum(item[0].shape[0] for item in rets)
        res = torch.zeros(total_batch_size, max_len, dtype=rets[0][0].dtype)
        res[:] = self.sample_params.eos_id
        sum_num_forwards = 0
        self.last_decode_seconds = max((item[2] for item in rets), default=None) if self.sample_params.polestar_profile_timing else None
        p = 0
        for item in rets:
            tensor, num_forwards = item[:2]
            sum_num_forwards = max(sum_num_forwards, num_forwards)
            out_len = int(tensor.shape[1])
            res[p:p+len(tensor), :out_len] = tensor
        self.num_forwards = sum_num_forwards
        return res

    def stop_serving(self):
        """ Stop model serving.
        """
        handle.stop_running()
        return None

    def port_selection(self, dp_size, start_port, end_port, max_retries=3):
        # select continuous port for each dp group ranging from start_port to end_port
        ports=[]
        for dp in range(dp_size):
            for retry in range(max_retries):
                try:
                    # start with 7 ports: 1 nccl port; 6 continus port for server_args: 1 server_args default port + 5 extra ports if dp_attention is available
                    # for simplity, we select 7 continuous_ports for each group
                    port, dp_ports_list = find_continuous_ports(num_ports=7, start_port=start_port, end_port=end_port)
                    ports.append(port) # record the first port for this group
                    logger.info("Allocated port range [%d-%d] (%d ports) for DP group %d", dp_ports_list[0], dp_ports_list[-1], len(dp_ports_list), dp)
                    break
                except ValueError as e:
                    if retry < max_retries - 1:
                        logger.warning("DP rank %d, attempt %d failed: %s. Retrying...",dp, retry + 1, e)
                        time.sleep(1)
                    else:
                        raise RuntimeError(
                            f"Failed to allocate ports for DP rank {dp} after {max_retries} attempts"
                        ) from e
        return ports
