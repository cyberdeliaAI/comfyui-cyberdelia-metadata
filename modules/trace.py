from collections import deque, defaultdict
from .defs.samplers import SAMPLERS
from .utils.log import print_warning

class Trace:
    _trace_cache = {}

    @staticmethod
    def _resolve_node_id(node_id, prompt):
        """Return the key used by *prompt* for a potential node id.

        ComfyUI serializes prompt node ids as strings, but some internal
        callers can still supply an integer.  Resolve that representation
        difference here without ever interpreting arbitrary input values as
        node ids.
        """
        if isinstance(node_id, bool) or not isinstance(node_id, (str, int)):
            return None

        if node_id in prompt:
            return node_id

        alternate_id = str(node_id) if isinstance(node_id, int) else None
        if alternate_id is not None and alternate_id in prompt:
            return alternate_id

        return None

    @classmethod
    def _linked_node_id(cls, value, prompt):
        """Extract the source node from one ComfyUI input link.

        API prompt links have the shape ``[node_id, output_index]``.  The
        second element is an output slot, not another node.  Other lists are
        literal input values and must not be traversed item by item.

        A small number of ComfyUI internals/custom nodes expose an equivalent
        mapping, so those are accepted too, while still validating that the
        referenced source actually exists in this prompt.
        """
        if isinstance(value, (list, tuple)):
            if (
                len(value) != 2
                or isinstance(value[1], bool)
                or not isinstance(value[1], int)
                or value[1] < 0
            ):
                return None
            return cls._resolve_node_id(value[0], prompt)

        if not isinstance(value, dict):
            return None

        link = value.get("link")
        if isinstance(link, (list, tuple)):
            return cls._linked_node_id(link, prompt)

        candidate = link
        if candidate is None:
            candidate = value.get("node_id")
        if candidate is None:
            candidate = value.get("id")
        return cls._resolve_node_id(candidate, prompt)

    @classmethod
    def _bfs_traverse(cls, start_node_id, prompt, visit_node, edge_condition=None):
        start_node_id = cls._resolve_node_id(start_node_id, prompt)
        if start_node_id is None:
            return

        Q = deque([(start_node_id, 0)])
        visited_nodes = set()
        visited_edges = set()

        while Q:
            current_node_id, distance = Q.popleft()
            if current_node_id in visited_nodes or current_node_id not in prompt:
                continue
            visited_nodes.add(current_node_id)

            node = prompt[current_node_id]
            visit_node(current_node_id, node, distance)

            for value in node.get("inputs", {}).values():
                next_id = cls._linked_node_id(value, prompt)
                if next_id is None:
                    continue

                edge = (current_node_id, next_id)
                if edge in visited_edges or (edge_condition and not edge_condition(current_node_id, next_id)):
                    continue

                visited_edges.add(edge)
                Q.append((next_id, distance + 1))

    @classmethod
    def trace(cls, start_node_id, prompt):
        resolved_start_node_id = cls._resolve_node_id(start_node_id, prompt)
        if resolved_start_node_id is None:
            return {}

        # A graph's node classes alone are not a safe cache identity: two
        # prompts can contain the same nodes/classes but connect them
        # differently, and two starts in a cycle can reach the same node set
        # at different distances.  Prompt identity + resolved start is exact
        # for the lifetime of a generation; hook.pre_execute clears the cache
        # before the next prompt is run.
        cache_key = (id(prompt), resolved_start_node_id)
        if cache_key in cls._trace_cache:
            return cls._trace_cache[cache_key]

        trace_tree = {}
        def build_trace(nid, node, dist):
            trace_tree[nid] = (dist, node.get("class_type", ""))
        cls._bfs_traverse(resolved_start_node_id, prompt, build_trace)
        cls._trace_cache[cache_key] = trace_tree
        return trace_tree

    @classmethod
    def find_node_by_class_types(cls, trace_tree, class_type_set, node_id=None):
        if node_id:
            node = trace_tree.get(node_id)
            if node and node[1] in class_type_set:
                return node_id
        else:
            for nid, (_, class_type) in trace_tree.items():
                if class_type in class_type_set:
                    return nid
        return None

    @classmethod
    def find_node_with_fields(cls, prompt, required_fields):
        for node_id, node in prompt.items():
            if required_fields & set(node.get("inputs", {}).keys()):
                return node_id, node
        return None, None
    
    @classmethod
    def find_all_nodes_with_fields(cls, prompt, required_fields):
        results = []
        for node_id, node in prompt.items():
            if required_fields & set(node.get("inputs", {}).keys()):
                results.append((node_id, node))
        return results

    @classmethod
    def find_sampler_node_id(cls, trace_tree):
        """Find the primary sampler node in the trace tree.

        When multiple samplers exist (e.g. 1st pass + upscale pass), the
        farthest one from the save node is the primary generation sampler
        whose settings (steps, cfg, seed, sampler) should be reported.
        """
        sampler_classes = set(SAMPLERS.keys())
        best_nid = None
        best_dist = -1
        for nid, (dist, class_type) in trace_tree.items():
            if class_type in sampler_classes and dist > best_dist:
                best_nid = nid
                best_dist = dist
        if best_nid:
            return best_nid
        print_warning("Could not find a sampler node in the trace tree!")

    @classmethod
    def filter_inputs_by_trace_tree(cls, inputs, trace_tree, prefer_nearest):
        filtered_inputs = defaultdict(list)
        for meta, input_list in inputs.items():
            for node_id, input_value in input_list:
                trace = trace_tree.get(node_id)
                if trace:
                    filtered_inputs[meta].append((node_id, input_value, trace[0]))

        for key in filtered_inputs:
            filtered_inputs[key].sort(key=lambda x: x[2], reverse=not prefer_nearest)  # nearest first if True

        return filtered_inputs
