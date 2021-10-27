

def save_html_page(body, fpath, title=None):
    html = '<html><head>'
    html += '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css”rel=”nofollow" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">'
    html += ''
    if title is not None:
        html += '<title>' + title + '</title>'
    html += '</head><body><div class="container">' + body + '</div></body></html>'

    with open(fpath, 'w') as f:
        f.write(html)
