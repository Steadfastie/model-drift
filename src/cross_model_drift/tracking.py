from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from cross_model_drift.config import AppConfig, load_config


@contextmanager
def clearml_task(
    task_name: str,
    *,
    config: AppConfig | None = None,
    task_type: str = "data_processing",
    tags: list[str] | None = None,
    reuse_last_task_id: bool = False,
    init: bool = True,
) -> Generator[Any, None, None]:
    """Create a ClearML Task when tracking is enabled.

    Notebooks can pass ``init=False`` to skip remote logging while keeping the
    same call sites. Credentials come from ``~/.clearml/clearml.conf`` after
    the local server is up.
    """
    if not init:
        yield None
        return

    from clearml import Task

    cfg = config or load_config()
    task = Task.init(
        project_name=cfg.clearml_project,
        task_name=task_name,
        task_type=Task.TaskTypes(task_type),
        reuse_last_task_id=reuse_last_task_id,
        auto_connect_frameworks=True,
    )
    if tags:
        task.add_tags(tags)
    try:
        yield task
    finally:
        task.close()


def log_metrics(task: Any, metrics: Mapping[str, float], title: str = "metrics") -> None:
    if task is None:
        return
    logger = task.get_logger()
    for name, value in metrics.items():
        if value is None:
            continue
        logger.report_single_value(name=f"{title}/{name}", value=float(value))


def log_parameters(task: Any, params: Mapping[str, Any], name: str = "params") -> None:
    if task is None:
        return
    task.connect(dict(params), name=name)
