from ipywidgets import HTML, HBox, Label
from IPython.display import display
import time


class Bar_Color:
    purple = f"""background: rgb(2,0,36);
            background: radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(121,9,101,1) 0%, rgba(121,9,101,1) 30%, rgba(3,24,28,1) 100%); border-radius: 1rem;"""
    green = f"""background: rgb(0,255,61);
            background: radial-gradient(circle, rgba(0,255,61,1) 0%, rgba(9,121,106,1) 0%, rgba(9,121,40,1) 19%, rgba(0,205,244,1) 100%)"""
    teal = f"""background: rgb(12,186,142);
            background: linear-gradient(90deg, rgba(12,186,142,1) 0%, rgba(12,176,134,1) 7%, rgba(72,230,189,1) 86%, rgba(148,244,219,1) 100%)"""
    green_gradient = f"""linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(29,239,253,1) 50%, rgba(252,176,69,1) 100%)"""
    cotton_candy = f"""background: rgb(238,174,202);
background: radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%)"""
    sunset = f"""background: rgb(34,193,195);
            background: linear-gradient(0deg, rgba(34,193,195,1) 0%, rgba(253,187,45,1) 100%)"""
    synthlite = f"""background-color: #00DBDE;
            background-image: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%)"""


def get_color(colour):
    if colour == "purple":
        colour = Bar_Color.purple
    elif colour == "green":
        colour = Bar_Color.green
    elif colour == "teal":
        colour = Bar_Color.teal
    elif colour == "green_gradient":
        colour = Bar_Color.green_gradient
    elif colour == "cotton_candy":
        colour = Bar_Color.cotton_candy
    elif colour == "sunset":
        colour = Bar_Color.sunset
    elif colour == "synthlite":
        colour = Bar_Color.synthlite
    else:
        colour = Bar_Color.green_gradient
    return colour


def iprogress(iterable, desc="Progress", colour=None):
    total = len(iterable)
    start_time = time.time()
    
    colour = get_color(colour)
    # Create the HTML widget
    progress_bar = HTML(f"""
    <style>
        .container {{
            margin: 10px auto;
            width: 500px;
            height: 15px;
            text-align: center;
        }}

        @keyframes progressAnimationStrike {{
            from {{ width: 0 }}
            to   {{ width: 100% }}
        }}

        .progress2 {{
            padding: 6px;
            border-radius: 30px;
            background: rgba(0, 0, 0, 0.25);  
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.25), 0 1px rgba(255,255,255,.08);
        }}

        .progress-bar2 {{
            height:18px; 
                border-radius:30px; 
                background-image:
                    linear-gradient(to bottom,
                        rgba(255,255,255,.3),
                        rgba(255,255,255,.05));
                transition-property:
                    width; 
                transition-duration:.4s; 
                transition-timing-function:cubic-bezier(.4,.61,.355,.955);
                animation-name:none; 
                animation-duration:.4s; 
                animation-timing-function:cubic-bezier(.4,.61,.355,.955);
                animation-fill-mode:both; 
                animation-delay:.4s; 
                animation-iteration-count:1; 
                animation-play-state:normal; 
                animation-direction:normal; 
                animation-name:none; 
                animation-duration:.4s; 
                animation-timing-function:cubic-bezier(.4,.61,.355,.955);
                animation-fill-mode:both; 
                animation-delay:.4s; 
                animation-iteration-count:1; 
                animation-play-state:normal; 
                animation-direction:normal;
        }}

        .progress-moved .progress-bar2 {{
            width:0%; 
            background-color:#EF476F;  
        }}
    </style>

    <div class="container">    
        <div class="progress2 progress-moved">
            <div class="progress-bar2" >
            </div>                       
        </div> 
    </div>
    """)

    # Create the description and postfix labels
    description = Label(value="Progress:")
    postfix = Label(value="Elapsed: 00:00 | Remaining: 00:00")

    # Display the progress bar and labels in a horizontal box
    display(HBox([description, progress_bar, postfix]))

    for i, item in enumerate(iterable):
        elapsed = time.time() - start_time
        elapsed = int(elapsed)
        percent = (i + 1) / total
        remaining = int((total - i - 1) * elapsed / (i + 1)) if i > 0 else 0
        it_per_sec = i / elapsed if elapsed > 0 else float('inf')
        
        # Update the progress bar value
        progress_bar.value = f"""
    <style>
        .container {{
            margin: 10px auto;
            width: 500px;
            height: 10px;
            text-align: center;
        }}

        @keyframes progressAnimationStrike {{
            from {{ width: 0 }}
            to   {{ width: {percent*100}% }}
        }}

        .progress2 {{
            padding: 3px;
            border-radius: 1rem;
            background: rgba(0, 0, 0, 0.25);  
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.25), 0 1px rgba(255,255,255,.08);
        }}

        .progress-bar2 {{
            height:10px; 
                border-radius:1rem; 
                background-image:
                    {colour};
                transition-property:
                    width; 
                transition-duration:.4s; 
                transition-timing-function:cubic-bezier(.4,.61,.355,.955);
                animation-name:none; 
                animation-duration:.4s; 
                animation-timing-function:cubic-bezier(.4,.61,.355,.955);
                animation-fill-mode:both; 
                animation-delay:.4s; 
                animation-iteration-count:1; 
                animation-play-state:normal; 
                animation-direction:normal;
        }}

        .progress-moved .progress-bar2 {{
            width:{percent*100}%; 
            background-color:#EF476F;  
        }}
    </style>

    <div class="container">    
        <div class="progress2 progress-moved">
            <div class="progress-bar2" >
            </div>                       
        </div> 
    </div>
    """
        
        # Update the postfix label
        description.value = f"{desc}: {int(percent*100)}%"
        postfix.value = f"Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d} | Remaining: {remaining // 60:02d}:{remaining % 60:02d} | Iterations/sec: {it_per_sec:.2f}"
        
        yield item
    progress_bar.close()
