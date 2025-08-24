"""Task Registry module for managing BciPy tasks.

This module provides a registry system for BCI tasks, allowing tasks to be
dynamically discovered and accessed by the GUI and command line tools.
User-defined tasks can be added to the Registry.
"""

from typing import Dict, List, Type, TypeVar

from bcipy.task import Task

# Type variable for Task subclasses
T = TypeVar('T', bound=Task)


class TaskRegistry:
    """Registry for managing and accessing BCI tasks.

    This class maintains a registry of all available task types in the system.
    It automatically discovers and registers all non-abstract Task subclasses
    when initialized, and provides methods for accessing and managing tasks.

    Attributes:
        registry_dict: Dictionary mapping task names to task classes.
    """

    def __init__(self) -> None:
        """Initialize the task registry.

        Collects all non-abstract subclasses of Task and registers them.
        Imports task modules to ensure all tasks are discovered.
        """
        # Import task modules to ensure all tasks are discovered
        from bcipy.task import actions  # noqa
        from bcipy.task.paradigm import matrix, rsvp, vep  # noqa

        self.registry_dict: Dict[str, Type[Task]] = {}
        self.collect_subclasses(Task)  # type: ignore[type-abstract]

    def collect_subclasses(self, cls: Type[T]) -> None:
        """Recursively collect and register non-abstract subclasses.

        Args:
            cls: The base class to collect subclasses from.

        Note:
            Subclasses are only registered if they have no abstract methods.
        """
        for sub_class in cls.__subclasses__():
            # Only register non-abstract subclasses
            if not getattr(sub_class, '__abstractmethods__', False):
                if hasattr(sub_class, 'name'):
                    self.registry_dict[sub_class.name] = sub_class
                else:
                    raise ValueError(f'Task class {sub_class} missing name attribute')
            self.collect_subclasses(sub_class)

    def get(self, task_name: str) -> Type[Task]:
        """Get a task class by its name.

        Args:
            task_name: Name of the task to retrieve.

        Returns:
            Type[Task]: The task class.

        Raises:
            ValueError: If the task name is not registered.
        """
        if task_name in self.registry_dict:
            return self.registry_dict[task_name]
        raise ValueError(f'{task_name} not a registered task')

    def get_all_types(self) -> List[Type[Task]]:
        """Get all registered task classes.

        Returns:
            List[Type[Task]]: List of all registered task classes.
        """
        return list(self.registry_dict.values())

    def list(self) -> List[str]:
        """Get names of all registered tasks.

        Returns:
            List[str]: List of registered task names.
        """
        return list(self.registry_dict.keys())

    def calibration_tasks(self) -> List[Type[Task]]:
        """Get all registered calibration tasks.

        Returns:
            List[Type[Task]]: List of registered calibration task classes.
        """
        from bcipy.task.calibration import BaseCalibrationTask
        return [task for task in self.get_all_types() if issubclass(task, BaseCalibrationTask)]

    def register_task(self, task: Type[Task]) -> None:
        """Register a new task with the registry.

        This method allows registration of additional tasks after initialization.
        Tasks imported during initialization are automatically registered.

        Args:
            task: The task class to register.
        """
        self.registry_dict[task.name] = task
