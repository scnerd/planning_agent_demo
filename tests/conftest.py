import logging

from rich.logging import RichHandler


def pytest_configure(config):
    """
    Configure logging with rich handler for all tests.
    Sets the root logger to WARNING level and planning_agent_demo logger to DEBUG level.
    """
    # Configure the root logger with a RichHandler at WARNING level
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # Set the planning_agent_demo logger to DEBUG level
    logger = logging.getLogger("planning_agent_demo")
    logger.setLevel(logging.DEBUG)
