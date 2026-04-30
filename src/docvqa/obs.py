def setup_observability():
    import logfire

    logfire.configure(console=False, inspect_arguments=False, scrubbing=False)
    logfire.instrument_litellm()
    logfire.instrument_dspy()
