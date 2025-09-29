def find_indexes(document, text):

    document = document.lower()
    text = text.lower()
    document = document.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    index_start, index_end = None, None

    # text_long_list = []
    # for letter in document:
    #     text_long_list.append(letter)


    for i in range(len(document)):
        if document[i:i+len(text)] == text:
            index_start = i
            index_end = i+len(text)
            break
    
            
    return(index_start,index_end)

#