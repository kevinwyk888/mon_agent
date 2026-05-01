def __getattr__(name: str):
    if name == "MonitoringAgent":
        from mon_agent.agent import MonitoringAgent
        return MonitoringAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MonitoringAgent"]

