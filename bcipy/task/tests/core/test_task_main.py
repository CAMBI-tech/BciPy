import unittest
from bcipy.task import Task, TaskData, TaskMode


class TestTask(unittest.TestCase):

    def test_task_fails_without_name(self):
        mode = TaskMode.CALIBRATION

        class TestTask(Task):

            def execute(self) -> TaskData:
                ...

        with self.assertRaises(AssertionError):
            TestTask(mode=mode)

    def test_task_fails_without_mode(self):
        name = "test task"

        class TestTask(Task):

            def execute(self) -> TaskData:
                ...

        with self.assertRaises(AssertionError):
            TestTask(name=name)

    def test_task_fails_without_execute(self):
        name = "test task"
        mode = TaskMode.CALIBRATION

        class TestTask(Task):
            pass

        with self.assertRaises(TypeError):
            TestTask(name=name, mode=mode)

    def test_task_initializes(self):
        name = "test task"
        mode = TaskMode.CALIBRATION

        class TestTask(Task):

            def __init__(self, name: str, mode: TaskMode):
                self.name = name
                self.mode = mode

            def execute(self) -> TaskData:
                ...
        task = TestTask(name=name, mode=mode)
        self.assertEqual(task.name, name)
        self.assertEqual(task.mode, mode)


if __name__ == '__main__':
    unittest.main()
