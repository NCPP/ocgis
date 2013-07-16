# Source: https://bitbucket.org/rajeev/django-piston/src/5f6a37c26d95/piston/templatetags/htmlemitter.py
from django import template

register = template.Library()

@register.filter('annotate')
def annotate(node):
    if isinstance(node, dict):
        s = '<dl>'
        for key,value in node.items():
            s += annotate((key, value))
        s += '</dl>'
        return s
    elif isinstance(node, tuple):
        return '<dt>'+node[0]+'</dt><dd>'+annotate(node[1])+'</dd>'
    elif isinstance(node, list):
        s = '<ul>'
        for item in node:
            s += '<li>'+annotate(item)+'</li>'
        s += '</ul>'
        return s
    else:
        return unicode(node)