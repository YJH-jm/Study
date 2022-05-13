def tag(name, *content, cls=None, **attrs):
    """하나 이상의 HTML tag 생성"""
    print(name, content, cls, attrs)
    
    print("bool(content) : ", bool(content))
    print("bool(attrs) : ",bool(attrs))
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                           for attr, value
                           in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' %
                         (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)