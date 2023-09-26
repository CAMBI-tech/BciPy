"""Useful for viewing computed positions in a window"""
import time
from typing import Callable, List, Tuple, Union

from psychopy import visual
from psychopy.visual.circle import Circle
from psychopy.visual.line import Line
from psychopy.visual.rect import Rect
from psychopy.visual.shape import ShapeStim

from bcipy.display.components.layout import (Layout, at_top, centered,
                                             envelope, height_units,
                                             scaled_size)
from bcipy.display.paradigm.vep.layout import BoxConfiguration, checkerboard


def make_window():
    """Make a sample window on which to draw."""
    return visual.Window(size=[700, 500],
                         fullscr=False,
                         screen=0,
                         waitBlanking=False,
                         color='black',
                         winType='pyglet')


def draw_boundary(win,
                  layout: Layout,
                  color: str = 'blue',
                  line_px: int = 5,
                  name: str = None):
    """Display the layout's outline."""
    if name:
        print(f"Drawing boundary for {name}")
    top = Line(win=win,
               units=layout.units,
               start=(layout.left, layout.top),
               end=(layout.right, layout.top),
               lineColor=color,
               lineWidth=line_px)

    bottom = Line(win=win,
                  units=layout.units,
                  start=(layout.left, layout.bottom),
                  end=(layout.right, layout.bottom),
                  lineColor=color,
                  lineWidth=line_px)

    left = Line(win=win,
                units=layout.units,
                start=(layout.left, layout.top),
                end=(layout.left, layout.bottom),
                lineColor=color,
                lineWidth=line_px)
    right = Line(win=win,
                 units=layout.units,
                 start=(layout.right, layout.top),
                 end=(layout.right, layout.bottom),
                 lineColor=color,
                 lineWidth=line_px)
    border = [top, right, bottom, left]
    for line in border:
        print(f"Drawing line from {line.start} to {line.end}")
        line.draw()


def draw_position(win: visual.Window,
                  pos: Tuple[float, float],
                  color: str = 'blue',
                  size: Union[float, Tuple[float, float]] = 0.025,
                  name: str = None):
    """Draw the provided positions"""
    if name:
        print(f"Drawing position {name} at {pos}")
    circle = Circle(win, pos=pos, color=color, size=size)
    circle.draw()


def draw_positions(win: visual.Window,
                   positions: List[Tuple[float, float]],
                   color: str = 'blue',
                   size: Union[float, Tuple[float, float]] = 0.025):
    """Draw the provided positions"""
    for pos in positions:
        draw_position(win, pos, color, size)


def show_layouts(window: visual.Window):
    """Show boundaries for various layouts"""
    draw_boundary(window, Layout(), 'red', 5, name='full_screen')
    draw_boundary(window,
                  at_top(window, 0.1),
                  color='blue',
                  line_px=5,
                  name='task_bar')
    draw_boundary(window,
                  centered(width_pct=0.9, height_pct=0.9),
                  color='green',
                  line_px=5,
                  name='main_display')

    window.flip()


def show_layout_coords(win: visual.Window):
    """Show the points that make up a layout's envelope"""
    layout = centered(width_pct=0.9, height_pct=0.9)
    layout.parent = win
    draw_boundary(win, layout, color='green')
    draw_position(win, (layout.left, layout.top),
                  size=layout.scaled_size(0.025),
                  name='top_left')
    draw_position(win, (layout.right, layout.bottom), name='bottom_right')

    Rect(win,
         pos=(layout.left, layout.top),
         size=layout.scaled_size(0.05),
         lineColor='blue').draw()
    Rect(win,
         pos=(layout.right, layout.bottom),
         size=layout.scaled_size(0.05),
         lineColor='blue').draw()
    win.flip()


def demo_vep(win: visual.Window):
    """Demo layout elements for VEP display"""
    layout = centered(width_pct=0.9, height_pct=0.9)
    layout.parent = win

    box_config = BoxConfiguration(layout, num_boxes=4, height_pct=0.28)
    size = box_config.box_size
    positions = box_config.positions

    draw_boundary(win, layout, color='gray', line_px=2)
    draw_positions(win,
                   positions,
                   color='blue',
                   size=layout.scaled_size(0.025))

    vep_colors = [('white', 'black'), ('red', 'green'), ('blue', 'yellow'),
                  ('orange', 'green'), ('white', 'black'), ('red', 'green')]
    colors = ['#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232']

    for i, pos in enumerate(positions):
        color_tup: Tuple[str, str] = vep_colors[i]
        board = checkerboard(squares=16,
                             colors=color_tup,
                             center=pos,
                             board_size=layout.scaled_size(0.2))
        for square in board:
            Rect(win,
                 pos=square.pos,
                 size=square.size,
                 lineColor='white',
                 fillColor=square.color).draw()

        rect = visual.TextBox2(win,
                               text=' ',
                               borderColor=colors[i],
                               pos=pos,
                               size=size)
        rect.draw()
    win.flip()


def demo_checkerboard(win: visual.Window):
    """Demo checkerboard"""
    board = checkerboard(squares=16,
                         colors=('red', 'green'),
                         center=(0, 0),
                         board_size=scaled_size(0.6, win.size))
    for square in board:
        Rect(win,
             pos=square.pos,
             size=square.size,
             lineColor='white',
             fillColor=square.color).draw()
    win.flip()


def demo_height_units(win: visual.Window):
    """Demo behavior when using height units"""
    norm_layout = centered(width_pct=0.85, height_pct=0.85)
    draw_boundary(win, norm_layout, line_px=5, color='red')
    layout = height_units(win.size)
    layout.resize_width(0.85)
    layout.resize_height(0.85)
    draw_boundary(win, layout, line_px=3, color='green')

    circle = Circle(win,
                    pos=layout.center,
                    color='blue',
                    size=0.025,
                    units='height')
    circle.draw()
    rects = [
        Rect(win,
             pos=layout.center,
             size=layout.scaled_size(0.05),
             lineColor='green',
             units='height'),
        # don't need to scale with height units
        Rect(win,
             pos=layout.left_top,
             size=(0.05, 0.05),
             lineColor='green',
             units='height'),
        # scaling the size should still work
        Rect(win,
             pos=layout.right_bottom,
             size=layout.scaled_size(0.05),
             lineColor='green',
             units=layout.units)
    ]
    for rect in rects:
        rect.draw()
    win.flip()


def demo_checkerboard2(win: visual.Window):
    """Demo checkerboard rendered using a complex shapestim"""
    board_pos = (0, 0)
    board_size = scaled_size(0.6, win.size)

    colors = ('red', 'green')
    board = checkerboard(squares=25,
                         colors=colors,
                         center=board_pos,
                         board_size=board_size)
    board_boundaries = envelope(pos=board_pos, size=board_size)

    evens = []
    odds = []
    for square in board:
        square_boundary = envelope(pos=square.pos, size=square.size)
        if square.color == colors[0]:
            evens.append(square_boundary)
        else:
            odds.append(square_boundary)

    rect = ShapeStim(win,
                     lineColor='green',
                     fillColor='green',
                     vertices=board_boundaries)
    rect.draw()
    stim = ShapeStim(win,
                     lineWidth=0,
                     fillColor=colors[0],
                     closeShape=True,
                     vertices=[board_boundaries, *evens])
    stim.draw()
    win.flip()


def run(demo: Callable[[visual.Window], None], seconds=30):
    """Run the given function for the provided duration.

    Parameters
    ----------
        demo - function to run; a Window object is passed to this function.
        seconds - Window is closed after this duration.
    """
    try:
        win = make_window()
        print(
            f'Displaying window for {seconds}s... (Interrupt [Ctl-C] to stop)\n'
        )

        demo(win)
        while True:
            time.sleep(seconds)
            break
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Demo stopped')
    except Exception as other_exception:
        print(f'{other_exception}')
        raise other_exception
    finally:
        win.close()
        print('Demo complete.')


run(demo_vep)
# run(demo_height_units)
# run(show_layout_coords)
# run(demo_checkerboard)
# run(demo_checkerboard2)
