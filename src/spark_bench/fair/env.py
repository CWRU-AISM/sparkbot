"""LIBERO env loading and the LIBERO-PRO perturbed suites.

The runner imports upstream LIBERO (which registers only the base suites) and reads each perturbed suite
from the LIBERO-PRO data folders that the LIBERO config (the folder LIBERO_CONFIG_PATH names) points at.
A missing perturbed file is an error: the runner never falls back to the unperturbed base suite.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re

import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

log = logging.getLogger(__name__)

# LIBERO init-state files hold pickled numpy arrays, which torch >= 2.6 refuses to load by default.
_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_full

PERTURBATION_SUFFIX = {'position': 'swap', 'task': 'task', 'language': 'lan', 'object': 'object'}


def get_perturbation_suite(base_suite: str, perturbation: str) -> str:
    """libero_object + position -> libero_object_swap."""
    return f"{base_suite}_{PERTURBATION_SUFFIX[perturbation]}"


def make_env(bddl_path: str, cfg):
    """The LIBERO env for one BDDL scene."""
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cfg.cam_height,
        camera_widths=cfg.cam_width,
        camera_depths=True,
        horizon=2000,  # the default 500 is too short for multi-step pick-place
    )


def load_libero_env(suite_name: str, task_id: int, cfg):
    """A base-suite LIBERO task: (env, task, init_states, bddl_path)."""
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return make_env(bddl_path, cfg), task, task_suite.get_task_init_states(task_id), bddl_path


def check_upstream_libero(suite: str) -> None:
    """Refuse a LIBERO package that registers the perturbed suites itself (the LIBERO-PRO fork): its
    registered suites build different scenes, and the protocol runs used upstream LIBERO."""
    if suite in benchmark.get_benchmark_dict():
        import libero
        raise SystemExit(f"{suite} is registered by the imported libero package ({list(libero.__path__)}); "
                         f"install upstream LIBERO and keep LIBERO-PRO only as the data folder")


_PRO_OBJECTS_DONE = False


def _register_pro_objects() -> None:
    """Register the object classes LIBERO-PRO adds (the object perturbation swaps in e.g. 'black_bowl')
    into upstream LIBERO. Runs LIBERO-PRO's own object modules from <libero root>/envs/objects, so their
    assets resolve under LIBERO-PRO's folder; a class upstream already registers is kept."""
    global _PRO_OBJECTS_DONE
    if _PRO_OBJECTS_DONE:
        return
    import libero.libero.envs.base_object as bo
    objects_dir = os.path.join(os.path.dirname(get_libero_path('bddl_files')), 'envs', 'objects')
    if not os.path.isdir(objects_dir):
        raise FileNotFoundError(f"{objects_dir} missing: the LIBERO config must point at a full LIBERO-PRO "
                                f"checkout (libero/libero), not only its data folders")
    orig = bo.register_object

    def register_new(cls):
        key = "_".join(re.sub(r"([A-Z0-9])", r" \1", cls.__name__).split()).lower()
        bo.OBJECTS_DICT.setdefault(key, cls)
        return cls
    bo.register_object = register_new
    try:
        for f in ('google_scanned_objects', 'hope_objects', 'turbosquid_objects', 'articulated_objects'):
            path = os.path.join(objects_dir, f + '.py')
            spec = importlib.util.spec_from_file_location(f'_libero_pro_objects_{f}', path)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    finally:
        bo.register_object = orig
    _PRO_OBJECTS_DONE = True


class ProSuite:
    """A LIBERO-PRO perturbed suite read from ``bddl_files/<suite>`` and ``init_files/<suite>``. LIBERO-PRO
    keeps the base suite's task order and file names; the scene, instruction and init states come from the
    perturbed folder."""

    def __init__(self, base_suite: str, name: str):
        self.base = benchmark.get_benchmark_dict()[base_suite]()
        self.name, self.n_tasks = name, self.base.n_tasks
        for kind in ('bddl_files', 'init_states'):
            d = os.path.join(get_libero_path(kind), name)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"{d} missing: LIBERO-PRO data for {name} is not installed")
        _register_pro_objects()

    def get_task_bddl_file_path(self, i: int) -> str:
        return os.path.join(get_libero_path("bddl_files"), self.name, self.base.get_task(i).bddl_file)

    def get_task(self, i: int):
        t = self.base.get_task(i)
        lang = t.language
        with open(self.get_task_bddl_file_path(i)) as f:
            for line in f:
                if '(:language' in line:
                    lang = line.split('(:language', 1)[1].rsplit(')', 1)[0].strip()
                    break
        return t._replace(problem_folder=self.name, language=lang)

    def get_task_init_states(self, i: int):
        return torch.load(os.path.join(get_libero_path("init_states"), self.name,
                                       self.base.get_task(i).init_states_file))
