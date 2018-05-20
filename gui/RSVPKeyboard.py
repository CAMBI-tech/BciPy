from utility.bci_gui import BCIGui
import wx

# Start the app and init the main GUI
app = wx.App(False)
gui = BCIGui(title="RSVPKeyboard", size=(650, 650), background_color='black')

# STATIC TEXT!
gui.add_static_text(
    text='RSVPKeyboard', position=(175, 0), size=30, color='white')
gui.add_static_text(
    text='Please enter user ID:', position=(170, 70), size=15, color='white')
gui.add_static_text(
    text='Chose your experiment type:', position=(75, 350), size=15, color='white')

# BUTTONS!
gui.add_button(
    message="Calibration",
    position=(75, 400), size=(100, 100),
    color='red')
gui.add_button(
    message="Copy Phrase", position=(200, 400),
    size=(100, 100),
    color='blue')
gui.add_button(
    message="Copy Phrase C.", position=(325, 400),
    size=(100, 100),
    color='green')
gui.add_button(
    message="Free Spell", position=(450, 400),
    size=(100, 100),
    color='orange')

# TEXT INPUT
gui.add_text_input(position=(175, 100), size=(250, 25))

gui.add_image(
    path='./static/images/gui_images/ohsu.png', position=(5, 0), size=125)

gui.add_image(
    path='./static/images/gui_images/neu.png', position=(510, 0), size=125)


# Make the GUI Show now
gui.show_gui()
app.MainLoop()
