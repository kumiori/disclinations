from xvfbwrapper import Xvfb
from pyvista.plotting.utilities import xvfb
import pyvista
import time
try:
    xvfb.start_xvfb(wait=0.05)
    print("Xvfb started successfully.")
    vdisplay = Xvfb()
    vdisplay.start()
    time.sleep(1)
    vdisplay.stop()
    
except OSError as e:
    # If Xvfb fails to start, you can handle the error here
    print("Xvfb could not be started, proceeding without it.")
    print(f"Error details: {e}")

# Enable off-screen rendering
pyvista.OFF_SCREEN = True

# Your PyVista or visualization code here
