import fitz

#Parses the input string of page intervals and returns a list of tuples
def parse_page_intervals(interval_string, pagecount):
    try:
        
        # Split the string by commas to get each interval
        intervals = interval_string.split(',')
        tuple_list = []
        
        for interval in intervals:
            # Check if there's a hyphen (interval case) or just a single number
            if '-' in interval:
                start, end = map(int, interval.split('-'))
            else:
                # If no hyphen, treat the single number as both start and end
                start = end = int(interval)

            # Automatically swap start and end if start > end
            if start > end:
                start, end = end, start

            # Check if start and end are within valid range
            if start < 1 or end > pagecount:
                print(f"Invalid interval: {start}-{end}. Page numbers must be between 1 and {pagecount}.")
                return False
            
            # Add the tuple to the list
            tuple_list.append((start, end))

        # Return the list of tuples
        return tuple_list

    except ValueError:
        # Handle cases where conversion to integers fails
        print("Invalid input format. Please use comma-separated intervals with two integers separated by a hyphen.")
        return False

def generate_delete_intervals(keep_intervals,pagecount):
    # Sort the intervals to keep by start index
    delete_intervals = []
    
    # Handle the part before the first interval
    if keep_intervals[0][0] > 1:
        delete_intervals.append((1, keep_intervals[0][0] - 1))
    
    # Handle the gaps between consecutive intervals
    for i in range(1, len(keep_intervals)):
        prev_end = keep_intervals[i - 1][1]
        curr_start = keep_intervals[i][0]
        if curr_start-prev_end > 1:
            delete_intervals.append((prev_end + 1, curr_start - 1))
    
    # Handle the part after the last interval
    if keep_intervals[-1][1] < pagecount:
        delete_intervals.append((keep_intervals[-1][1] + 1, pagecount))
    
    return delete_intervals

#creates a selective copy of our input document-- copy the desired pages
def selectiongen(copy_tuple_list, doc):
    trimdoc=fitz.open()
    for tup in copy_tuple_list:
        trimdoc.insert_pdf(doc, from_page=(tup[0]-1),to_page=(tup[1]-1))
    return trimdoc

#deletes pages that were not selected -- modifies the input doc directly
def selectioncut(del_tuple_list, doc):
    expanded_tuple = tuple(i-1 for start, end in del_tuple_list for i in range(start, end + 1))
    doc.delete_pages(expanded_tuple)
    return doc


# extracts / collates selected pages only from an input PDF.
def selection(pagelist_string,doc):
    pagecount=doc.page_count
    tuple_list=parse_page_intervals(pagelist_string,pagecount)

    #if any intervals overlap, we create a copy with duplicated pages
    sorted_intervals = sorted(tuple_list, key=lambda x: x[0])
    for i in range(len(sorted_intervals) - 1):
        if sorted_intervals[i][1]>=sorted_intervals[i + 1][0]:
            return selectiongen(tuple_list,doc)

    #if no intervals overlap, we can avoid copying the pdf and delete the undesired pages in place
    #note that this does not preserve the order of intervals specified in the pagelist_string
    del_tuple_list=generate_delete_intervals(sorted_intervals,pagecount)
    return selectioncut(del_tuple_list,doc)