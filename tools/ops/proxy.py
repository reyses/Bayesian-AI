import http.server
import urllib.request

class Proxy(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        url = 'http://127.0.0.1:11434' + self.path
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length)
        
        req = urllib.request.Request(url, data=body, method='POST')
        for k, v in self.headers.items():
            if k.lower() != 'host':
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for k, v in response.headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        url = 'http://127.0.0.1:11434' + self.path
        req = urllib.request.Request(url, method='GET')
        for k, v in self.headers.items():
            if k.lower() != 'host':
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for k, v in response.headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

if __name__ == '__main__':
    server = http.server.ThreadingHTTPServer(('0.0.0.0', 11435), Proxy)
    print("Proxying 0.0.0.0:11435 -> 127.0.0.1:11434")
    server.serve_forever()
