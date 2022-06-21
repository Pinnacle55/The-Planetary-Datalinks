#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import time
from collections import Counter
from scipy.stats import entropy
# import random
# from tqdm.notebook import tqdm

# t_start = time.perf_counter()
# pokedex = pd.read_csv('Squirdle_Pokedex.csv')

# my_dex = pokedex[["name", "generation", "type1", "type2", "height", "weight"]]

# #Stolen data cleaning code from Jonny#
# # my_dex["name"] = my_dex["name"].str.lower()
# my_dex.weight.fillna(my_dex.weight.max(), inplace = True)

# #Encoding values for types
# type_dict = {'water':'1','normal':'2','grass':'3','bug':'4','psychic':'5','fire':'6',
#              'electric':'7','rock':'8','dark':'9','poison':'10','dragon':'11','ghost':'12',
#              'fighting':'13','ground':'14','ice':'15','steel':'16','fairy':'17',
#              'flying':'18','none':'19'}


# #Initialising variables
# poke_gen = np.array(my_dex["generation"])

# #Mapping Pokemon types to numerical values
# type1_map=pd.Series(my_dex['type1']).map(type_dict)
# type2_map=pd.Series(my_dex['type2']).map(type_dict)

# #poke_type1 = np.array(my_dex["type1"])
# #poke_type2 = np.array(my_dex["type2"])
# poke_type1 = np.array(type1_map).astype(np.uint8)
# poke_type2 = np.array(type2_map).astype(np.uint8)

# poke_height = np.array(my_dex["height"])
# poke_weight = np.array(my_dex["weight"])

# #Pokemon Generation

# """
# OUTPUT GEN CODES
# 1 - LESS THAN, 2 - EQUAL, 3 - GREATER
# """

# # This creates a 1072 x 1072 boolean array with True/False depending on whether its greater or 
# # less than a specific generation
# # Basically, if you do an inequality of an array of (1072,1) against an array of (1,1072), you'll
# # get an boolean array of (1072,1072) 
# gen_greater_mask = poke_gen[:,None] > poke_gen
# gen_less_mask = poke_gen[:,None] < poke_gen
# gen_eq_mask = poke_gen[:,None] == poke_gen

# # Converts the masks to arrays of numbers instead
# # When gen is less, it gives 1, when gen is equal, it gives 2, etc.
# gen_less_encode = gen_less_mask.astype(np.uint8)
# gen_eq_encode = (gen_eq_mask*2).astype(np.uint8)
# gen_greater_encode = (gen_greater_mask*3).astype(np.uint8)

# # Flattens the array to a single layer by selecting the maximum value amongst the three arrays
# gens_encoded = np.maximum.reduce([gen_less_encode,gen_eq_encode,gen_greater_encode])



# #Pokemon Types 

# """
# OUTPUT TYPE CODES
# 0 - MISS, 1 - MISPLACED, 2- HIT
# """

# type1_hit = poke_type1[:,None] == poke_type1
# type2_hit = poke_type2[:,None] == poke_type2

# type1_misplace = poke_type1[:,None] == poke_type2
# type2_misplace = poke_type2[:,None] == poke_type1

# type1_misplace_encode = type1_misplace.astype(np.uint8)
# type1_hit_encode = (type1_hit*2).astype(np.uint8)
# type1_encoded = np.maximum(type1_misplace_encode,type1_hit_encode)

# type2_misplace_encode = type2_misplace.astype(np.uint8)
# type2_hit_encode = (type2_hit*2).astype(np.uint8)
# type2_encoded = np.maximum(type2_misplace_encode,type2_hit_encode)


# """
# OUTPUT HEIGHT CODES
# 1 - LESS THAN, 2 - EQUAL, 3 - GREATER
# """

# #Pokemon Heights
# height_greater_mask = poke_height[:,None] > poke_height
# height_less_mask = poke_height[:,None] < poke_height
# height_eq_mask = poke_height[:,None] == poke_height


# height_less_encode = height_less_mask.astype(np.uint8)
# height_eq_encode = (height_eq_mask*2).astype(np.uint8)
# height_greater_encode = (height_greater_mask*3).astype(np.uint8)

# height_encoded = np.maximum.reduce([height_less_encode,height_eq_encode,height_greater_encode])

# """
# OUTPUT WEIGHT CODES
# 1 - LESS THAN, 2 - EQUAL, 3 - GREATER
# """
# #Pokemon Weights
# weight_greater_mask = poke_weight[:,None] > poke_weight
# weight_less_mask = poke_weight[:,None] < poke_weight
# weight_eq_mask = poke_weight[:,None] == poke_weight


# weight_less_encode = weight_less_mask.astype(np.uint8)
# weight_eq_encode = (weight_eq_mask*2).astype(np.uint8)
# weight_greater_encode = (weight_greater_mask*3).astype(np.uint8)

# weight_encoded = np.maximum.reduce([weight_less_encode,weight_eq_encode,weight_greater_encode])


# stacked_final_dex = np.stack([gens_encoded,type1_encoded,type2_encoded,height_encoded,weight_encoded])

class SquirdleSolver():
    """
    A Class that keeps track of the guesses and patterns
    and handles the calculation of the optimal guesses at any point in time.
    
    .from_csv()
    Loads a .csv database of Pokemon stats. 
    Requires ["name", "generation", "type1", "type2", "height", "weight"]
    """
    def __init__(self, pokedex, matrix):
        
        self.pokedex = pokedex
        self.matrix = matrix
        self.possible_pokemon = np.array(list(range(len(self.matrix[0]))))
        
    @classmethod
    def from_csv(cls, filename):
        pokedex = pd.read_csv(filename)
        
        try:
            my_dex = pokedex[["name", "generation", "type1", "type2", "height", "weight"]]
        except:
            print("The .csv file must contain the following columns: ")
            print('"name", "generation", "type1", "type2", "height", and "weight"')

        # Replaces all unknown weights with the max weight (basically only Eternamax)
        my_dex.weight.fillna(my_dex.weight.max(), inplace = True)

        #Encoding values for types
        # This speeds up the calculation of the original matrix.
        type_dict = {'water':'1','normal':'2','grass':'3','bug':'4','psychic':'5','fire':'6',
                     'electric':'7','rock':'8','dark':'9','poison':'10','dragon':'11','ghost':'12',
                     'fighting':'13','ground':'14','ice':'15','steel':'16','fairy':'17',
                     'flying':'18','none':'19'}


        #Initialising variables
        poke_gen = np.array(my_dex["generation"])

        #Mapping Pokemon types to numerical values
        type1_map=pd.Series(my_dex['type1']).map(type_dict)
        type2_map=pd.Series(my_dex['type2']).map(type_dict)

        #poke_type1 = np.array(my_dex["type1"])
        #poke_type2 = np.array(my_dex["type2"])
        poke_type1 = np.array(type1_map).astype(np.uint8)
        poke_type2 = np.array(type2_map).astype(np.uint8)

        poke_height = np.array(my_dex["height"])
        poke_weight = np.array(my_dex["weight"])

        #Pokemon Array Generation

        """
        OUTPUT GEN CODES
        0 - WRONG, 1 - CORRECT, 2 - GREATER THAN, 3 - LESS THAN, 4 - WRONG POSITION
        """

        # This creates a 1072 x 1072 boolean array with True/False depending on whether its greater or 
        # less than a specific generation
        # Basically, if you do an inequality of an array of (1072,1) against an array of (1,1072), you'll
        # get an boolean array of (1072,1072) 
        gen_greater_mask = poke_gen[:,None] > poke_gen
        gen_less_mask = poke_gen[:,None] < poke_gen
        gen_eq_mask = poke_gen[:,None] == poke_gen

        # Converts the masks to arrays of numbers instead
        # See the above code in the """triple quotations"""
        gen_less_encode = (gen_less_mask*3).astype(np.uint8)
        gen_eq_encode = gen_eq_mask.astype(np.uint8)
        gen_greater_encode = (gen_greater_mask*2).astype(np.uint8)

        # Flattens the array to a single layer by selecting the maximum value amongst the three arrays
        gens_encoded = np.maximum.reduce([gen_less_encode,gen_eq_encode,gen_greater_encode])



        #Pokemon Types 

        """
        OUTPUT GEN CODES
        0 - WRONG, 1 - CORRECT, 2 - GREATER THAN, 3 - LESS THAN, 4 - WRONG POSITION
        """

        type1_hit = poke_type1[:,None] == poke_type1
        type2_hit = poke_type2[:,None] == poke_type2
        
        # Source of the big BUG!
        type1_misplace = poke_type2[:,None] == poke_type1
        type2_misplace = poke_type1[:,None] == poke_type2

        type1_misplace_encode = (type1_misplace*4).astype(np.uint8)
        type1_hit_encode = type1_hit.astype(np.uint8)
        type1_encoded = np.maximum(type1_misplace_encode,type1_hit_encode)

        type2_misplace_encode = (type2_misplace*4).astype(np.uint8)
        type2_hit_encode = type2_hit.astype(np.uint8)
        type2_encoded = np.maximum(type2_misplace_encode,type2_hit_encode)


        """
        OUTPUT GEN CODES
        0 - WRONG, 1 - CORRECT, 2 - GREATER THAN, 3 - LESS THAN, 4 - WRONG POSITION
        """

        #Pokemon Heights
        height_greater_mask = poke_height[:,None] > poke_height
        height_less_mask = poke_height[:,None] < poke_height
        height_eq_mask = poke_height[:,None] == poke_height


        height_less_encode = (height_less_mask*3).astype(np.uint8)
        height_eq_encode = height_eq_mask.astype(np.uint8)
        height_greater_encode = (height_greater_mask*2).astype(np.uint8)

        height_encoded = np.maximum.reduce([height_less_encode,height_eq_encode,height_greater_encode])

        """
        OUTPUT GEN CODES
        0 - WRONG, 1 - CORRECT, 2 - GREATER THAN, 3 - LESS THAN, 4 - WRONG POSITION
        """
        #Pokemon Weights
        weight_greater_mask = poke_weight[:,None] > poke_weight
        weight_less_mask = poke_weight[:,None] < poke_weight
        weight_eq_mask = poke_weight[:,None] == poke_weight


        weight_less_encode = (weight_less_mask*3).astype(np.uint8)
        weight_eq_encode = weight_eq_mask.astype(np.uint8)
        weight_greater_encode = (weight_greater_mask*2).astype(np.uint8)
        
        weight_encoded = np.maximum.reduce([weight_less_encode,weight_eq_encode,weight_greater_encode])
        
        # Stack the encoded arrays in the Squirdle order
        stacked_final_dex = np.stack([gens_encoded,type1_encoded,type2_encoded,height_encoded,weight_encoded])
        
        # Create an instance of Squirdle Solver.
        return SquirdleSolver(my_dex, stacked_final_dex)
        
    def get_entropy_from_matrix(self, 
                                poke1):
        """
        This gets entropy by using the matrix_array that we have as a look-up table. 
        Vectorised to be as efficient as possible, since we are calculating this extremely often.

        Because we're using a matrix, we need to convert the names of pokemon to numbers 
        in order to access the elements in the the array.

        poke1 - integer
        """
        # This finds the most common squirdle pattern when comparing the target pokemon (poke1) to the
        # possible pokemon
        # np.unique finds unique rows (axis = 0), and returns their counts in a separate array
        # We actually don't need the unique values - we just need the value_counts matrix
        probability_distribution = np.unique(self.matrix[:, poke1, self.possible_pokemon].T, 
                                             axis = 0, 
                                             return_counts = True)

        # The entropy function takes in a list of probabilities (values from 0-1)
        # and returns the entropy of the distribution, which is a value that describes how 
        # "certain" a result will be.
        entropy_value = entropy(probability_distribution[1], base = 2)

        return entropy_value
    
    def get_optimal_guess(self, num_results):
        """
        This finds the optimal next guess from a list of pokemon
        
        See if this can be vectorised.
        """
        list_of_entropies = list()

        # Finds the entropies of all pokemon in the list
        for pokemon in self.possible_pokemon:
            list_of_entropies.append(self.get_entropy_from_matrix(pokemon))
        
        # Note that our list of entropies is of the possible pokemon, so we need to retrieve
        # the element that has the highest entropy from the list of possible pokemon, not the 
        # original list

        # CHECK: "-" should be in front of np.asarray
        optimal_pokemon_index = self.possible_pokemon[np.argsort(-np.array(list_of_entropies))]

        # Returns the names of the most informative pokemon
        up_to = min(len(optimal_pokemon_index), num_results)
        return np.array(self.pokedex.name[optimal_pokemon_index][:up_to])
    
    def update_possible_pokemon(self,
                                guess, 
                                pattern):
        """
        What is the set of possible pokemon for the given guess-pattern combination?
        
        guess - string
        pattern - must be converted to the appropriate pattern
        """
        # Get the index number of the guessed pokemon
        poke_index = self.pokedex.index[self.pokedex["name"] == guess].to_list()

        # Extract the list of all possible results for the guessed pokemon from the
        # lookup matrix.
        # Find all indexes that match the given pattern.
        matches = np.argwhere(np.sum(self.matrix.T[poke_index][0] == pattern, axis = 1) > 4)

        # Remember that we may already have information from previous guesses,
        # so we need to find the intersection between our matches and the set
        # of allowed_pokemon to find the set of all pokemon the Secret Pokemon
        # could be
        possible_matches = np.intersect1d(matches.flatten(), np.array(self.possible_pokemon))
        
        # Update the set of possible_pokemon
        self.possible_pokemon = possible_matches
        
    def reset_possible_pokemon(self):
        """
        Reset the list of possible pokemon back to the full set of pokemon.
        """
        self.possible_pokemon = np.array(list(range(len(self.matrix[0]))))


# In[18]:


import tkinter as tk
from tkinter import ttk # Submodule for additional widgets
from PIL import ImageTk # For images

# NOTE: If you get a pyimage doesn't exist error, restart the kernel - this is an error associated
# with the anaconda distribution


class SquirdleApp(tk.Tk):
    """
    The actual tkinter app that will run the solver
    Requires an instance of the SquirdleSolver class
    """
    def __init__(self, solver: SquirdleSolver):
        super().__init__()
        self.solver = solver
        # Used for dynamic height management.
        self.height = 450
        
        # configure the root window
        self.title('Squirdle Solver')
        self.rowconfigure(0, minsize = 50, weight = 1)
        self.columnconfigure(1, minsize = 250, weight = 1)
        self.geometry("910x450")
        
        # Create the frame that holds all the patterns
        self.patternframe = PatternFrame(self, borderwidth = 2, relief = tk.GROOVE)
        self.patternframe.grid(row = 0, column = 1, pady = 20)
        
        # Create the frame that holds the recommendations/optimal guesses
        self.recommendframe = RecommendationsFrame(self, borderwidth = 2, relief = tk.GROOVE)
        self.recommendframe.grid(row = 1, column = 1, pady = 20)
        
        # Create the frame with a searchable selection box
        self.selectframe = PokemonSelectFrame(self, self.solver.pokedex["name"])
        self.selectframe.grid(row = 0, column = 2, rowspan = 2)
        
        # Create a button that allows you to reset the app for multiple plays
        self.resetbutton = tk.Button(self, text = "Reset", width = 20, 
                                     font = ("Helvetica", 14), command = self.reset_app)
        self.resetbutton.grid(row = 2, column = 1, columnspan = 2, pady = 20)
        
    def reset_app(self):
        """
        Function that resets the app. There must be a better way to do this.
        
        Possibly just
        SquirdleApp(self.solver)? <- This creates an entirely new window 
        That said, it doesn't require all of the reset code.
        
        self.__init__() <- This doesn't work either - creates a new window because
        we call super().__init__()
        """
        for widget in self.winfo_children():
            widget.destroy()
        
        ResultFrame.all_results = list()
        
        self.patternframe = PatternFrame(self, borderwidth = 2, relief = tk.GROOVE)
        self.patternframe.grid(row = 0, column = 1, pady = 20)
        self.recommendframe = RecommendationsFrame(self, borderwidth = 2, relief = tk.GROOVE)
        self.recommendframe.grid(row = 1, column = 1, pady = 20)
        
        self.selectframe = PokemonSelectFrame(self, self.solver.pokedex["name"])
        self.selectframe.grid(row = 0, column = 2, rowspan = 2)
        
        self.resetbutton = tk.Button(self, text = "Reset", width = 20, 
                                     font = ("Helvetica", 14), command = self.reset_app)
        self.resetbutton.grid(row = 2, column = 1, columnspan = 2, pady = 20)
        self.solver.reset_possible_pokemon()
        
        
class PokemonSelectFrame(tk.Frame):
    """
    The frame that holds the listbox and the searchable box.
    """
    def __init__(self, root, names, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        self.names = names
        
        # Button that submits the guess to the program.
        self.button = tk.Button(self, text = "Guess!", font = ("Helvetica", 16), 
                                width = 15, command = self.master.patternframe.create_result)
        self.button.grid(row = 2, pady = 10)
        
        # The entry box
        self.entry = tk.Entry(self, font = ("Helvetica", 16), width = 30)
        self.entry.grid(row = 3, sticky = "nsew")
        # Bind keyrelease to a function that reduces the possible names
        self.entry.bind("<KeyRelease>", self.update_selection)
        
        # The listbox that contains all pokemon names
        self.listbox = tk.Listbox(self, font = ("Helvetica", 16), width = 30, exportselection = False)
        self.listbox.grid(row = 4, sticky = "nsew")
        # Note that any function called by bind will pass an "event" argument
        # The only action that can be bound to listbox is ListboxSelect
        self.listbox.bind("<<ListboxSelect>>", self.select_item)
        
        self.refresh_selection()
        
    def refresh_selection(self, names = None):
        """
        Repopulates the listbox based on a given list.
        """
        # Delete all elements in the listbox
        self.listbox.delete(0, tk.END)
        
        # refresh_selection called without arguments should just put everything into the box
        if names is None:
            names = self.names
        
        for name in names:
            self.listbox.insert(tk.END, name)
            
    def select_item(self, event):
        """
        Puts whatever you click in the listbox into the entry box.
        """
        # This deletes the original stuff in the listbox
        self.entry.delete(0, tk.END)
        
        # This inserts the selected item into the listbox
        # Weirdly, ANCHOR works with clicking, but ACTIVE works with arrow keys
        # Current implementation works with clicks but NOT with arrow keys
        # self.entry.insert(0, self.listbox.get(tk.ANCHOR))
        
        # THIS WORKS WITH BOTH! VERY NICE
        if self.listbox.curselection() is not None:
            self.entry.insert(0, self.listbox.get(self.listbox.curselection()[0]))
        
    def update_selection(self, event):
        """
        Repopulates the listbox based on the contents of the entry box.
        """
        
        # Get what we typed
        typed = self.entry.get()
        
        # If the box is empty, then set back to the full list of pokemon names
        if typed == "":
            self.refresh_selection()
        # Else, search for the appropriate pokemon
        else:
            names = []
            for name in self.names:
                # If typed appears anywhere in the pokemon name
                if typed.lower() in name.lower():
                    names.append(name)
        
        self.refresh_selection(names)
    
    
class PatternFrame(tk.Frame):
    """
    The Class that will hold the rotating buttons of the guesses.
    It's responsible for passing these patterns to the Recommendations Frame.
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        # The order of the picture rotation.
        self.rotation = ["wrong", "correct", "up", "down", "wrongpos"]
        self.index = 0
        self.image_dict = {}
        
        # Inserts images into an image_dict
        for index, item in enumerate(self.rotation):
            self.image_dict[index] = ImageTk.PhotoImage(file = "Squirdle Pictures/"+item+".png")
        
    def create_result(self):
        """
        Get the text in the entrybox from the SelectFrame
        """
        # Dynamically adjusts the height of the app as more guesses are added.
        self.master.geometry(f"910x{self.master.height+(50*self.index+1)}")
        
        # If the entrybox is empty, do not add anything. 
        # Might want to adjust this to include incomplete words. There shouldn't be (people should
        # select from the listbox) but you never know.
        # Should just be X not in self.master.solver.pokedex["name"]
        if self.master.selectframe.entry.get() == "":
            return
        else:
            # Add the guess from the entrybox
            guessed_pokemon = self.master.selectframe.entry.get()
            # Create an appropriate ResultFrame
            self.result = ResultFrame(self, self.image_dict, guessed_pokemon)
            self.result.grid(row = self.index, column = 0, sticky = "nsew")
            self.index += 1
        
class ResultFrame(tk.Frame):
    """
    The Frame that holds the actual rotating buttons and labels.
    """
    # A Class-level attribute that will hold ALL instances of ResultFrame. 
    # We'll need this because we want to take all guesses into account.
    all_results = []
    
    def __init__(self, root, image_dict, guessed_pokemon, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        self.guessed_pokemon = guessed_pokemon
        self.image_dict = image_dict
        # Keep track of all instances of ResultFrame
        ResultFrame.all_results.append(self)
        
        # For type changes
        self.status_dict = {0: 1, 1: 4, 4: 0}
        
        # Makes it so that the guesses are nicely formatted and have enough space.
        self.columnconfigure(6, minsize = 30, weight = 1)
        
        # Unfortunately, no real way of looping through these since each button has a
        # unique meaning.
        
        # Generation Button
        self.generationbutton = RotatingButton(self, image = image_dict[0], borderwidth = 0,
                                         command = self.generation_change)
        self.generationbutton.grid(row = 1, column = 1)
        
        # Type 1 Button
        self.type1button = RotatingButton(self, image = image_dict[0], borderwidth = 0,
                                         command = self.type1_change)
        self.type1button.grid(row = 1, column = 2)
        
        # Type 2 Button
        self.type2button = RotatingButton(self, image = image_dict[0], borderwidth = 0,
                                         command = self.type2_change)
        self.type2button.grid(row = 1, column = 3)
        
        # Height Button
        self.heightbutton = RotatingButton(self, image = image_dict[0], borderwidth = 0,
                                         command = self.height_change)
        self.heightbutton.grid(row = 1, column = 4)
        
        # Weight Button
        self.weightbutton = RotatingButton(self, image = image_dict[0], borderwidth = 0,
                                         command = self.weight_change)
        self.weightbutton.grid(row = 1, column = 5)
        
        # If the guess has a long name (generally because its a regional form)
        # We split the name so that so that it appears shorter in the results frame.
        if " - " in self.guessed_pokemon:
            name = "\n".join(self.guessed_pokemon.split(" - "))
        else:
            name = self.guessed_pokemon
        self.guessedpokemon = tk.Label(self, text = name, font = ("Helvetica", 14))
        self.guessedpokemon.grid(row = 1, column = 6, padx = 10)
        
    def generation_change(self):
        """
        Function that controls how the generation button rotates. 
        Also refreshes the recommendations with each click.
        
        Rest of the functions are the same.
        """
        current_status = self.generationbutton.status
        if current_status > 2:
            current_status = 0
        else:
            current_status += 1
        self.generationbutton.config(image = self.image_dict[current_status])
        self.generationbutton.status = current_status
        
        #self.master = PatternFrame, self.master.master = Window
        self.master.master.recommendframe.refresh_recommendations()
        
    def height_change(self):
        current_status = self.heightbutton.status
        if current_status > 2:
            current_status = 0
        else:
            current_status += 1
        self.heightbutton.config(image = self.image_dict[current_status])
        self.heightbutton.status = current_status
        self.master.master.recommendframe.refresh_recommendations()
        
    def weight_change(self):
        current_status = self.weightbutton.status
        if current_status > 2:
            current_status = 0
        else:
            current_status += 1
        self.weightbutton.config(image = self.image_dict[current_status])
        self.weightbutton.status = current_status
        self.master.master.recommendframe.refresh_recommendations()
        
    def type1_change(self):
        current_status = self.type1button.status
        current_status = self.status_dict[current_status]
        self.type1button.config(image = self.image_dict[current_status])
        self.type1button.status = current_status
        self.master.master.recommendframe.refresh_recommendations()
        
    def type2_change(self):
        current_status = self.type2button.status
        current_status = self.status_dict[current_status]
        self.type2button.config(image = self.image_dict[current_status])
        self.type2button.status = current_status
        self.master.master.recommendframe.refresh_recommendations()
        
    def get_pattern(self):
        """
        Return the list of button statuses for all Buttons in the resultframe.
        """
        # This list looks at all the children of this instance of ResultsFrame.
        # If its a button, we grab its status.
        return [i.status for i in self.winfo_children() if i.winfo_class() == "Button"]
    

# We need to create a class of button so that we can get the STATUS of that button
class RotatingButton(tk.Button):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.status = 0
        self.master = root
    
class RecommendationsFrame(tk.Frame):
    """
    The Frame that will update to give the user the best guess for the selected patterns.
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        
        # The initial recommendations.
        self.recommendation_label = tk.Label(self, text = "Make a guess!", font = ("Helvetica", 12))
        self.recommendation_label.grid(row = 0, column = 0, padx = 50)
    
    def refresh_recommendations(self):
        """
        Collects the patterns from all ResultsFrames and uses them to narrow down the list of 
        possible pokemon.
        """
        # Reset the list of possible pokemon whenever we click - if we don't reset, our clicks
        # will eventually lead to impossible combinations. 
        self.master.solver.reset_possible_pokemon()
        
        # For each resultframe we have:
        for frame in ResultFrame.all_results:
            # Get its pattern (list) and name (str) and use them to update the possible pokemon
            self.master.solver.update_possible_pokemon(frame.guessed_pokemon, 
                                                       frame.get_pattern())
        
        # Get the 6 best guesses
        guess_list = self.master.solver.get_optimal_guess(6)
        
        # Destroy the original recommendation labels
        self.recommendation_label.destroy()
        
        # Create a new frame with a new set of labels.
        self.recommendation_label = UpdatingLabelsFrame(self, guess_list)
        self.recommendation_label.grid(row = 0, column = 0, padx = 50)

class UpdatingLabelsFrame(tk.Frame):
    """
    This creates the list of labels based on the optimal guesses as calculated from the given
    guess-pattern combinations.
    """
    def __init__(self, root, guess_list, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        
        # Some fluff
        self.fluff = tk.Label(self, text = "You should try guessing:", font = ("Helvetica", 14))
        self.fluff.grid(row = 0, column = 0)
        
        # If there are no legal pokemon
        if not guess_list.size > 0:
            self.label = tk.Label(self, text = "No legal Pokemon with these results.", font = ("Helvetica", 14))
            self.label.grid(row = 1, column = 0)
        # Otherwise, print the six best guesses
        else:
            for index, name in enumerate(guess_list):
                self.label = tk.Label(self, text = name, font = ("Helvetica", 14, 'bold'))
                self.label.grid(row = index + 1, column = 0)


# In[19]:


app = SquirdleApp(SquirdleSolver.from_csv("Squirdle_Pokedex.csv"))
app.mainloop()


# In[4]:


# from PIL import ImageTk
# import tkinter as tk

# window = tk.Tk()

# image_dict = dict()

# rotation = ["wrong", "correct", "up", "down", "wrongpos"]

# for index, item in enumerate(rotation):
#     image_dict[index] = ImageTk.PhotoImage(file = "Squirdle Pictures/"+item+".png")

# frame = ResultFrame(window, image_dict, "test")
# frame.grid()
# button = tk.Button(frame, image = image_dict[0])
# button.grid()


# # frame = ResultFrame(window, image_dict, "test")
# # frame.pack()

# window.mainloop()


# In[5]:


# test = SquirdleSolver.from_csv("Squirdle_Pokedex.csv")


# In[6]:


# test.update_possible_pokemon("Bulbasaur", [1,1,1,1,1])

# test.possible_pokemon


# In[ ]:




