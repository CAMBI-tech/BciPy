import wx
from gui.gui_main import BCIGui


# Start the app and init the main GUI
app = wx.App(False)
gui = BCIGui(title="Brain Computer Interface", size=(700, 400), background_color='black')

# STATIC TEXT!
gui.add_static_text(
    text='Brain Computer Interface', position=(125, 0), size=20, color='white')

# BUTTONS!
gui.add_button(
    message="RSVP",
    position=(125, 200), size=(100, 100),
    color='red')
gui.add_button(
    message="Matrix", position=(250, 200),
    size=(100, 100),
    color='blue')
gui.add_button(
    message="Shuffle", position=(375, 200),
    size=(100, 100),
    color='green')

gui.add_image(
    path='./static/images/gui_images/ohsu.png', position=(5, 0), size=125)

gui.add_image(
    path='./static/images/gui_images/neu.png', position=(550, 0), size=125)


# Make the GUI Show now
gui.show_gui()
app.MainLoop()
