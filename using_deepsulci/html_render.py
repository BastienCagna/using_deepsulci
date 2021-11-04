import os.path as op
import re
from enum import Enum
from collections.abc import Iterable

# TODO: Test all
# TODO: Add heritance between templates
# TODO: Implement if
# TODO: use var_name.sub_var_name => var_name[sub_var_name]


class BlockType(Enum):
    Unknown = 0
    IF = 1
    IF_END = -1
    FOR = 2
    FOR_END = -2


def apply_pipe(value, pipe):
    parts = pipe.split('(')
    function = parts[0].strip()
    if len(parts) > 1:
        params = list(s.strip() for s in parts[1].split(')')[0].split(','))
    else:
        params = []

    if function == 'ucfirst':
        return value.capitalize()
    if function == 'uppercase':
        return value.upper()
    if function == 'lowercase':
        return value.lower()
    if function == 'capitalize':
        return ' '.join(s.capitalize() for s in str(value).split(' '))
    if function == 'date':
        raise NotImplementedError()
    if function == "is_iterable":
        return isinstance(value, Iterable)

    raise ValueError("Unknown pipe function {}".format(function))


def apply_pipes(value, pipes):
    for pipe in pipes:
        value = apply_pipe(value, pipe)
    return value


def infer_block_type(tag: str) -> BlockType:
    in_tag = list(s.strip().lower() for s in tag.split(' ')[1:-1])
    tag_name = in_tag[0]
    if tag_name == 'for':
        if len(in_tag) != 4 or not in_tag[2] == 'in':
            raise SyntaxError("'for' block is badly defined.")
        return BlockType.FOR
    elif tag_name == 'if':
        return BlockType.IF
    elif tag_name == 'endfor':
        return BlockType.FOR_END
    elif tag_name == 'endif':
        return BlockType.IF_END
    return BlockType.Unknown


def block_childs(blocks: list, parent: dict):
    """ Add childs blocks in the parent block """
    # Search childs for each block
    # Assume that blocks are ordered by ascending start
    parent['childs'] = []
    for child in blocks[parent['id']+1:]:
        if child['start'] > parent['end']:
            break
        elif child['depth'] > (parent['depth'] + 1):
            continue
        else:
            child = block_childs(blocks, child)
            child['start'] -= parent['start']
            parent['childs'].append(child)
    return parent


def replace_block_by_tag(template, blocks, block):
    block['template'] = template[block['start']:block['end']].strip()
    block_len = block['end'] - block['start']
    template = template[:block['start']] + \
        '{{ @{:d} }}'.format(block['id']) + template[block['end']:]
    for b in blocks:
        if b['start'] > block['start']:
            b['start'] -= block_len
        if b['end'] > block['end']:
            b['end'] -= block_len


def replace_tags(template: str, data: dict) -> str:
    # Other tags
    for tag in re.findall(r'\{\{[a-z 0-9_\|\(\)\@]+\}\}', template):
        pipes = list(s.strip() for s in tag[2:-2].split('|'))
        varname = pipes[0]
        if varname in data:
            value = apply_pipes(data[varname], pipes[1:])
        else:
            value = ""
        template = template.replace(tag, value)
    return template


def replace_in_childs(template, data, parent):
    for child in parent['childs']:
        rendered_child = ""
        if child['type'] == BlockType.FOR:
            local_data = data.copy()
            in_tag = list(s.strip().lower()
                          for s in child['tag'].split(' ')[1:-1])
            for item in data[in_tag[3]]:
                local_data[in_tag[1]] = item
                rendered_child += replace_in_childs(
                    child['template'], local_data, child)
        if child['type'] == BlockType.IF:
            raise NotImplementedError()
        data['@{:d}'.format(child['id'])] = rendered_child
    template = replace_tags(template, data)
    return template


def render(template: str, data: dict, save_in=None):
    if template.endswith('.html'):
        with open(template, 'r') as f:
            template = "".join(f.readlines())

    # List block tags and get the depth of each tag
    block_tags = []
    r = re.compile("{\%[a-z 0-9_\|\(\)]+\%}")
    depth = 0
    max_depth = 0
    for id, block_tag in enumerate(r.finditer(template)):
        btype = infer_block_type(block_tag.group())
        reduce_depth = False
        if btype == BlockType.Unknown:
            continue
        elif btype in [BlockType.FOR, BlockType.IF]:
            depth += 1
            if depth > max_depth:
                max_depth = depth
        else:
            reduce_depth = True

        tag = block_tag.group()
        block_tags.append({
            'id': id,
            'tag': tag,
            'start': block_tag.start(),
            'type': btype,
            'depth': depth
        })

        if reduce_depth:
            depth -= 1

    # Match blocks start and end
    blocks = []
    for id, block_tag in enumerate(block_tags):
        block = block_tag
        for end_block_tag in block_tags[id+1:]:
            if end_block_tag['depth'] != block['depth'] or end_block_tag['start'] < block['start']:
                continue
            if block['type'] == BlockType.IF and end_block_tag['type'] == BlockType.IF_END \
                    or block['type'] == BlockType.FOR and end_block_tag['type'] == BlockType.FOR_END:
                block['start'] = block['start'] + len(block['tag'])
                block['end'] = end_block_tag['start']
                blocks.append(block)
                break
    del block_tags

    # Replace all block (starting by deepest ones) by tags ({%% [block_id] %%})
    # and store inner content in template property (to render it later)
    depth = max_depth
    while depth > 0:
        for block in blocks:
            if block['depth'] == depth:
                replace_block_by_tag(template, blocks, block)
        depth -= 1

    # Construct block hierarchy
    template_block = {'id': -1, 'tag': '', 'start': 0,
                      'type': BlockType.Unknown, 'depth': 0, 'end': len(template)}
    block_childs(blocks, template_block)
    print(blocks)
    print(template_block)
    template = replace_in_childs(template, data, template_block)

    if save_in:
        with open(save_in, 'w') as f:
            f.write(template)
    return template
