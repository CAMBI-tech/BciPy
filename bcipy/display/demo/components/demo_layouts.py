"""Useful for viewing computed positions in a window"""
import time
from itertools import cycle
from typing import Callable, List, Tuple, Union

from psychopy import visual
from psychopy.visual.circle import Circle
from psychopy.visual.grating import GratingStim
from psychopy.visual.line import Line
from psychopy.visual.rect import Rect

from bcipy.display.components.layout import (Layout, at_top, centered,
                                             height_units)
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


def vep_box_quadrant(win: visual.Window, pos: Tuple[float, float], anchor: str,
                     color: Tuple[str, str], size: Tuple[float, float]):

    pattern1 = GratingStim(win=win,
                           name=f'2x2-1-{pos}-{anchor}',
                           tex=None,
                           pos=pos,
                           size=size,
                           sf=1,
                           phase=0.0,
                           color=color[0],
                           colorSpace='rgb',
                           opacity=1,
                           texRes=256,
                           interpolate=True,
                           depth=-1.0,
                           anchor=anchor)
    pattern2 = GratingStim(win=win,
                           name=f'2x2-2-{pos}-{anchor}',
                           tex=None,
                           pos=pos,
                           size=size,
                           sf=1,
                           phase=0.0,
                           color=color[1],
                           colorSpace='rgb',
                           opacity=1,
                           texRes=256,
                           interpolate=True,
                           depth=-1.0,
                           anchor=anchor)
    border = Rect(win, pos=pos, size=size, lineColor='white', anchor=anchor)
    return [pattern1, pattern2, border]


def demo_vep(win: visual.Window):
    """Demo layout elements for VEP display"""
    layout = centered(width_pct=0.9, height_pct=0.9)
    layout.parent = win

    box_config = BoxConfiguration(layout, num_boxes=4, height_pct=0.28)
    size = box_config.box_size()
    positions = box_config.positions

    draw_boundary(win, layout, color='gray', line_px=2)
    draw_positions(win,
                   positions,
                   color='blue',
                   size=layout.scaled_size(0.025))

    vep_colors = [('white', 'black'), ('red', 'green'), ('blue', 'yellow'),
                  ('orange', 'green'), ('white', 'black'), ('red', 'green')]
    colors = ['#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232']
    # draw boxes
    anchor_points = ['right_bottom', 'left_bottom', 'left_top', 'right_top']

    for i, pos in enumerate(positions):
        gradient_colors: Tuple[str, str] = vep_colors[i]

        for anchor, color in zip(
                anchor_points,
                cycle([gradient_colors,
                       tuple(reversed(gradient_colors))])):
            for stim in vep_box_quadrant(win, pos, anchor, color,
                                         layout.scaled_size(0.1)):
                stim.draw()

        rect = visual.TextBox2(win,
                               text=' ',
                               borderColor=colors[i],
                               pos=pos,
                               size=size)
        rect.draw()
    win.flip()


def demo_checkerboard(win: visual.Window):
    """Demo checkerboard"""
    layout = centered(width_pct=0.9, height_pct=0.9)
    layout.parent = win
    board = checkerboard(layout,
                         squares=16,
                         colors=('red', 'green'),
                         center=(0, 0),
                         board_size=layout.scaled_size(0.4))
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


# run(demo_vep)
# run(demo_height_units)
# run(show_layout_coords)
run(demo_checkerboard)
