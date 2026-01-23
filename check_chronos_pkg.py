
try:
    import chronos
    print("chronos package found.")
    from chronos import ChronosPipeline
    print("ChronosPipeline imported.")
except ImportError:
    print("chronos package NOT found.")
except Exception as e:
    print(f"Error: {e}")
