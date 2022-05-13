def clip(text:str, max_len:"int > 0" = 80) -> str: # 함수 선언에 에너테이션 추가
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(" ", max_len)
            if space_after >= 0:
                end = space_after
    if end is None:
        end = len(text)
        
    return text[:end].rstrip()