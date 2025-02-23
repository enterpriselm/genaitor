import cProfile
import pstats
import io
import asyncio
from examples.advanced_usage import main  # Import the main function from your existing script

def profile():
    pr = cProfile.Profile()
    pr.enable()  # Start profiling

    asyncio.run(main())  # Run the main function

    pr.disable()  # Stop profiling
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())  # Print the profiling results

if __name__ == "__main__":
    profile() 