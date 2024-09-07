import fitz

#Parses an input string of page intervals
# Then extracts / collates those pages only from an input PDF.

def parse_page_intervals(interval_string, pdf_document):
    try:
        # Get the total number of pages in the document
        total_pages = pdf_document.page_count
        
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
            if start < 1 or end > total_pages:
                print(f"Invalid interval: {start}-{end}. Page numbers must be between 1 and {total_pages}.")
                return False
            
            # Add the tuple to the list
            tuple_list.append((start, end))

        # Return the list of tuples
        return tuple_list

    except ValueError:
        # Handle cases where conversion to integers fails
        print("Invalid input format. Please use comma-separated intervals with two integers separated by a hyphen.")
        return False


def selectioncut(string, doc):
	trimdoc=fitz.open()
	tuple_list=parse_page_intervals(string,doc)
	for tup in tuple_list:
		trimdoc.insert_pdf(doc, from_page=(tup[0]-1),to_page=(tup[1]-1))
	return trimdoc