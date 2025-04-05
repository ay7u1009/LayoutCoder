def read_json_file(path):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(path, data, is_np=False, is_box=False):
    import json
    import numpy as np

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    class ReprJSONEncoder(json.JSONEncoder):
        def default(self, obj):

            # 将指定对象转换为其 __repr__ 表示形式, isinstance失效
            if obj.__class__.__name__ == 'Box':
                return repr(obj)
            return super(ReprJSONEncoder, self).default(obj)

    kwargs = {
        "obj": data,
        "indent": 4,
        "ensure_ascii": False
    }

    if is_np:
        kwargs["cls"] = NpEncoder

    if is_box:
        kwargs["cls"] = ReprJSONEncoder

    with open(path, 'w') as f:
        kwargs["fp"] = f
        json.dump(**kwargs)