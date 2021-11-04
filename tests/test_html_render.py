from using_deepsulci import html_render
import os.path as op


def test_tag_rendering():
    rendered = html_render.render("{{ firstname | ucfirst }} {{ lastname | uppercase }}", {
                                  'firstname': 'basTien', 'lastname': 'cagna'})
    assert rendered == "Bastien CAGNA"


def test_block():
    rendered = html_render.render("{% for item in items %} {{ item }} {% endfor %}", {
                                  'items': ['un', 'deux', 'trois']})
    assert rendered == "un deux trois"

# def test_rendering():
#     test_data = {
#         "title": "test page",
#         "navigation": [
#             {'title': 'home', 'link': '#', 'is_active': True},
#             {'title': 'footer', 'link': '#footer'}
#         ],
#         "content": '<h1>This is a test page</h1><p>Nothing to do actually because this is a test.</p>',
#         "footer": '<ul><li>Item 1</li><li>Item 2</li></ul>'
#     }
#     html_render.render(
#         op.abspath(op.join(__file__, "..", "..", "templates", "index.html")), test_data)
