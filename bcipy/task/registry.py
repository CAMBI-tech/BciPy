"""Task Registry ; used to provide task options to the GUI and command line
tools. User defined tasks can be added to the Registry."""
from typing import Dict, List, Type
from bcipy.task import Task


class TaskRegistry:
    registry_dict: Dict[str, Type[Task]]

    def __init__(self):
        # Collects all non-abstract subclasses of Task. type ignore is used to work around a mypy bug
        # https://github.com/python/mypy/issues/3115
        from bcipy.task.paradigm import vep, rsvp, matrix  # noqa
        from bcipy.task import actions  # noqa

        self.registry_dict = {}
        self.collect_subclasses(Task)  # type: ignore[type-abstract]

    def collect_subclasses(self, cls: Type[Task]):
        """Recursively collects all non-abstract subclasses of the given class and adds them to the registry."""
        for sub_class in cls.__subclasses__():
            if not getattr(sub_class, '__abstractmethods__', False):
                self.registry_dict[sub_class.name] = sub_class
            self.collect_subclasses(sub_class)

    def get(self, task_name: str) -> Type[Task]:
        """Returns a task type based on its name property."""
        if task_name in self.registry_dict:
            return self.registry_dict[task_name]
        raise ValueError(f'{task_name} not a registered task')

    def get_all_types(self) -> List[Type[Task]]:
        """Returns a list of all registered tasks."""
        return list(self.registry_dict.values())

    def list(self) -> List[str]:
        """Returns a list of all registered task names."""
        return list(self.registry_dict.keys())

    def calibration_tasks(self) -> List[Type[Task]]:
        """Returns a list of all registered calibration tasks."""
        from bcipy.task.calibration import BaseCalibrationTask
        return [task for task in self.get_all_types() if issubclass(task, BaseCalibrationTask)]

    def register_task(self, task: Type[Task]) -> None:
        """Registers a task with the TaskRegistry."""
        # Note that all imported tasks are automatically registered when the TaskRegistry is initialized. This
        # method allows for the registration of additional tasks after initialization.
        self.registry_dict[task.name] = task
