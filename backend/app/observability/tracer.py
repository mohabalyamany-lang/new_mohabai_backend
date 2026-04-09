import time
import json


class Tracer:

    def start(self, name):
        return {"name": name, "start": time.time()}

    def end(self, span, data=None):
        span["duration"] = time.time() - span["start"]
        span["data"] = data or {}
        print(json.dumps(span))


tracer = Tracer()
