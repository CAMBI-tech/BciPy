# Web server for editing parameters.json file
from os import sep
import os.path
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
from collections import defaultdict
import urllib
from typing import Dict, Any, Callable
import threading
import time
import logging


def page(load_from, save_to, msg=None, err=None):

    msg_element = ""
    if msg:
        msg_element = f"<div class='info bg-success'>{msg}</div>"

    err_element = ""
    if err:
        err_element = f"<div class='info bg-danger text-danger'>{err}</div>"
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device_width, initial-scale=1">
<meta name="Description" content="Form to edit BCI configuration parameters">
<title>BCI Parameters</title>
<link rel="stylesheet" href="assets/css/bootstrap.min.css.gz"/>
    <style>
    .container-fluid {{ max-width: 798px; }}
    .help-block {{ color:#555; }}
    .info {{
        padding: 10px;
        border: 0.5px solid lightgray;
        margin-bottom: 12px;
    }}
    body, .form-control {{ font-size: 16px; }}
    </style>
    <script type='text/javascript'>
      function stop() {{
         console.log('calling stop function');
         var xmlhttp = new XMLHttpRequest();
         xmlhttp.onreadystatechange = function() {{
             if (this.readyState == 4 && this.status == 200) {{
                 window.close();
             }}
         }};
         xmlhttp.open('GET', '/stop', true);
         xmlhttp.send();
         return true;
      }}
    </script>
</head>
<body>
    <div class='container-fluid'>
        <h1>BCI Configuration</h1>
        {err_element}
        {msg_element}
        {close_link()}
        {params_form(load_from, save_to)}
        {form(load_from, save_to)}
    </div>
</body>
</html>"""


def bootstrap():
    p = sep.join(['.', 'gui', 'assets', 'css', 'bootstrap.min.css'])

    with open(p, 'r') as f:
        return str(f.read())


def close_page():
    return f"""<!DOCTYPE html>
    <html lang='en'>
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device_width, initial-scale=1">
    <meta name="Description" content="Form to edit BCI configuration parameters">
    <title>BCI Parameters</title>
        <style>
        {bootstrap()}
        .container-fluid {{ max-width: 798px; }}
        .help-block {{ color:#555; }}
        .info {{
            padding: 10px;
            border: 0.5px solid lightgray;
            margin-bottom: 12px;
        }}
        body, .form-control {{ font-size: 16px; }}
        </style>

    </head>
    <body>
        <div class='container-fluid'>
            <h1>BCI Configuration</h1>
            <div class='info bg-success'>Please close this browser window.</div>
        </div>
    </body>
    </html>"""


def label(key, param):
    lbl = f"<label for='{key}'>{param['readableName']}</label>"
    if param['readableName'] != param['helpTip']:
        lbl += f"<p class='help-block small'>{param['helpTip']}</p>"

    return lbl


def bool_input(key, param):
    checked = "checked='checked'" if param['value'] == "true" else ""
    return (f"<label><input type='checkbox' id='{key}' name='{key}' {checked}> "
            f"{param['readableName']}"
            "</label>"
            f"<p class='help-block small'>{param['helpTip']}</p>")


def numeric_input(key, param):
    return (f"{label(key, param)}"
            "<input class='form-control' type='number' step='any' "
            f"name='{key}' id='{key}' value='{param['value']}' />")


def params_file_input(load_from, save_to):
    key = 'params_load_file'
    helpTip = ('To load parameters from a different file, change the path here'
               ' and click the <em>Load</em> link. You must then Submit the '
               f'form to save to: {save_to}.')
    return text_input('params_load_file', {'value': load_from,
                                           'readableName': "Parameters loaded from",
                                           'helpTip': helpTip})


def text_input(key, param):
    return (f"{label(key, param)}"
            f"<input class='form-control' type='text' name='{key}' "
            f"id='{key}' value='{param['value']}' />")


def hidden_input(key, value):
    return (f"<input class='form-control' type='hidden' name='{key}' "
            f"id='{key}' value='{value}' />")


def select_option(val, selected_value):
    s = "selected='selected'" if val == selected_value else ""
    return f"<option value='{val}' {s}>{val}</option>"


def selection_input(key, param):
    options = [select_option(opt, param['value'])
               for opt in param['recommended_values']]

    return (f"{label(key, param)}"
            f"<select class='form-control' name='{key}' id='{key}'>"
            f"{options}"
            "</select>")


def input(key, param):
    if param['type'] == "bool":
        return bool_input(key, param)
    elif type(param['recommended_values']) == list:
        return selection_input(key, param)
    elif param['type'] in ['float', 'int']:
        return numeric_input(key, param)
    else:
        return text_input(key, param)


def form_section(header, inputs):
    f_inputs = '\n'.join(inputs)
    return (f"<h3>{header}</h3>"
            "<div class='section well' >"
            f"{f_inputs}"
            "</div>")


def form_input(key: str, param):
    return f"<div class='form-group'>{input(key, param)}</div>"


def params_form(json_file: str, save_to: str) -> str:
    """Form that alows a user to load a different parameters file."""

    content = (
        f"<form name='change-input' method='POST' action='/load'>"
        "<div class='form-group'>"
        f"{params_file_input(json_file, save_to)}"
        "<button type='submit' class='btn btn-link'>Load</button>"
        "</div>"
        f"</form>")
    return content


def close_link():
    return "<a class='btn btn-link' href='/stop'>Close</a>"


def form(json_file: str, save_to: str) -> str:
    """Creates a web form from a json file"""

    with open(json_file) as f:
        data = f.read()
    params = json.loads(data)

    # group inputs by section
    sections = defaultdict(list)
    for k, v in params.items():
        sections[v['section']].append(form_input(k, v))

    items = [form_section(section, form_inputs)
             for section, form_inputs in sections.items()]

    inputs = '\n'.join(items)
    content = ("<form name='bci_parameters' method='POST' action='/save'>"
               f"{hidden_input('filename', save_to)}"
               f"{inputs}"
               "<button type='submit' class='btn btn-default'>Save</button>"
               f"{close_link()}"
               "</form>")

    return content


def request_handler_builder(load_file: str, save_to: str=None,
                            stop_server: Callable[[], None]=None):
    """Constructs a WebServer with the provided parameters. This factory design
    pattern is needed since HTTPServer creates a new Handler for every
    request."""

    class RequestHandler(BaseHTTPRequestHandler, object):
        def __init__(self, *args, **kwargs):
            self.load_file = load_file
            self.save_to = save_to
            self.stop_server = stop_server
            if self.save_to is None:
                self.save_to = load_file
            super(RequestHandler, self).__init__(*args, **kwargs)

        def send_headers(self, mime_type="text/html", encoding=False):
            self.send_response(200)
            self.send_header("Content-type", mime_type)
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()

        def send_content(self, content, mime_type="text/html"):
            self.send_headers(mime_type)
            self.wfile.write(content.encode("utf-8"))

        def send_gzip(self, content, mime_type="text/css"):
            self.send_headers(mime_type, encoding="gzip")
            self.wfile.write(content)

        def get_post_data(self) -> Dict[str, str]:
            length = int(self.headers['Content-Length'])
            return urllib.parse.parse_qs(
                self.rfile.read(length).decode('utf-8'))

        # @override
        def do_GET(self):
            """Handles GET requests"""
            logging.debug(f"Path: {self.path}")
            if self.path == "/":
                content = page(self.load_file, self.save_to)
                self.send_content(content)
            elif self.path == '/assets/css/bootstrap.min.css.map':
                p = sep.join(['.', 'gui', 'assets', 'css',
                              'bootstrap.min.css.map'])
                try:
                    with open(p, 'rb') as f:
                        self.send_headers("application/json")
                        self.wfile.write(f.read())
                    return
                except IOError:
                    self.send_error(
                        404, "File Not Found: {}".format(p))
            elif self.path == '/assets/css/bootstrap.min.css.gz':
                p = sep.join(['.', 'gui', 'assets', 'css',
                              'bootstrap.min.css.gz'])
                try:
                    with open(p, 'rb') as f:
                        self.send_gzip(f.read(), mime_type="text/css")
                    return
                except IOError:
                    self.send_error(
                        404, "File Not Found: {}".format(p))
            elif self.path == '/stop':
                self.stop()
            else:
                self.send_error(404, "Not Found")

        # @override
        def do_POST(self):
            """Handles POST requests."""
            if self.path == "/save":
                self.save()
            elif self.path == '/load':
                self.load()

        def load(self):
            """Loads a new parameters file from the POSTed path."""
            post_data = self.get_post_data()
            new_file = post_data['params_load_file'][0]

            try:
                self.send_content(page(new_file, self.save_to))
            except Exception as e:
                err = f"Error loading from file: {new_file}; {str(e)}"
                self.send_content(page(self.load_file, self.save_to, err=err))

        def save(self):
            """Saves the form to the parameters.json file."""

            post_data = self.get_post_data()

            # Read all json data from existing load_file.
            with open(self.load_file) as f:
                data = f.read()
            params = json.loads(data)

            # Overwrite value fields from form-submitted data.
            for k, v in post_data.items():
                if k in params:
                    # TODO: escape inputs?
                    if params[k]['value'] in ["false", "true"]:
                        val = "true" if v[0] == "on" else "false"
                    else:
                        val = v[0]
                    params[k]['value'] = val

            # Write to disk.
            filename = post_data['filename'][0]
            with open(filename, 'w') as outfile:
                json.dump(params, outfile, indent=4)
            self.send_content(page(filename, self.save_to,
                                   msg="Successfully saved!"))

        def stop(self):
            self.send_content(close_page())

            if self.stop_server:
                self.stop_server()

    return RequestHandler


class Signal(object):
    running = True


def start_params_server(load_file="parameters/parameters.json", save_to=None,
                        host='', port=8080):

    signal = Signal()

    def stop_server():
        signal.running = False

    httpd = HTTPServer((host, port), request_handler_builder(
        load_file, save_to, stop_server=stop_server))
    logging.debug(time.asctime(), "Server Started - %s:%s" % (host, port))

    try:
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        webbrowser.open("http://{}:{}".format(host, port))

        while signal.running:
            time.sleep(0.5)

        httpd.shutdown()

    except KeyboardInterrupt:
        httpd.server_close()

    httpd.server_close()
    logging.debug(time.asctime(), "Server Stopped - %s:%s" % (host, port))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-file', default='parameters/parameters.json',
                        help='load data from this file')
    parser.add_argument('--save-to', help='save data to this file')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8080)

    args = parser.parse_args()
    logging.debug(args)
    start_params_server(save_to=args.save_to, load_file=args.load_file,
                        host=args.host, port=args.port)
