def justify_row(row, indent=4, right_edge=80):
    edge = right_edge - (indent + 1)
    words = row.strip().split(' ')
    if len(words) == 1:
        tabbedrows = words
    elif any([len(word) > (right_edge - indent) for word in words]):
        tabbedrows = [row]
    else:
        tabbedrows = []
        done = False
        while True:
            new_row = []
            words_idx = 0
            while len(' '.join(new_row)) <= edge:
                try:
                    new_row.append(words[words_idx])
                    words_idx += 1
                except IndexError:
                    done = True
                    break
            if not done and len(' '.join(new_row)) != edge:
                new_row.pop(-1)
                words_idx -= 1
            tabbedrows.append(' '.join(new_row))
            if done:
                break
            else:
                words = words[words_idx:]
    tabbedrows = [' ' * indent + ii for ii in tabbedrows]
    return (tabbedrows)
