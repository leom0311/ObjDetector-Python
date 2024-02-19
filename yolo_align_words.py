import os
import argparse
from typing import Dict, List, Set, Tuple
from PIL import Image
import math

from .yolo_dataset import *

class AnnotatedFrame:
    objects     : List[AnnotatedObject]
    words       : List[List[AnnotatedObject]]

    def __init__(self):
        super().__init__()

        self.objects    = list()
        self.words      = list()
    
    def to_list(self) -> List[AnnotatedObject]:
        result = list(self.objects)
        for word in self.words:
            result += word
        return result


def group_card_suit_rank(
    frame               : AnnotatedFrame,
    card_class          : int,
    partial_card_class  : int,
    rank_classes        : Set[int],
    suit_classes        : Set[int]):

    words_list = list()

    while True:
        new_word    : List[AnnotatedObject] = list()

        # find card object
        best_fit_index : int = -1
        for i, obj in enumerate(frame.objects):
            if obj.class_id == card_class:
                best_fit_index = i
                break
            elif obj.class_id == partial_card_class:
                best_fit_index = i
                break
        if best_fit_index < 0:
            break

        card = frame.objects.pop(best_fit_index)
        new_word.append(card)
        
        left    = card.cx - card.width / 2
        right   = card.cx + card.width / 2
        top     = card.cy - card.height / 2
        bottom  = card.cy + card.height / 2

        if card.class_id == card_class:
            right = left + (right - left) / 3
        
        # find rank objects that belong to this card
        region_left     = left  - (right - left) *  0.2
        region_right    = right + (right - left) *  0.2
        region_top      = top   + (bottom - top) * -0.1
        region_bottom   = top   + (bottom - top) *  0.4

        rank_chars : List[AnnotatedObject] = list()
        i = 0
        while i < len(frame.objects):
            obj = frame.objects[i]
            if (obj.class_id in rank_classes and
                obj.cx - obj.width/2 >= region_left and obj.cx + obj.width/2 <= region_right and
                obj.cy - obj.height/2 >= region_top and obj.cy + obj.height/2 <= region_bottom):

                rank_chars.append(obj)
                frame.objects.pop(i)
            else:
                i += 1
        
        # align rank chars
        if len(rank_chars) > 1:
            rank_chars.sort(key = lambda o : o.cx)
            mean_cy     = sum([o.cy for o in rank_chars]) / len(rank_chars)
            mean_height = sum([o.height for o in rank_chars]) / len(rank_chars)

            split_pos : List[float] = list()
            split_pos.append(rank_chars[0].cx - rank_chars[0].width/2)
            for i in range(0, len(rank_chars)-1):
                split_pos.append((
                    rank_chars[i].cx + rank_chars[i].width/2 +
                    rank_chars[i+1].cx - rank_chars[i+1].width/2
                ) / 2)
            split_pos.append(rank_chars[-1].cx + rank_chars[-1].width/2)

            for i, obj in enumerate(rank_chars):
                rank_chars[i].cx        = (split_pos[i] + split_pos[i+1]) / 2
                rank_chars[i].width     = split_pos[i+1] - split_pos[i]
                rank_chars[i].cy        = mean_cy
                rank_chars[i].height    = mean_height

        new_word += rank_chars
        
        # find suit objects that belong to this card
        region_top      = top   + (bottom - top) * 0.3
        region_bottom   = top   + (bottom - top) * 0.7

        i = 0
        while i < len(frame.objects):
            obj = frame.objects[i]
            if (obj.class_id in suit_classes and
                obj.cx - obj.width/2 >= region_left and obj.cx + obj.width/2 <= region_right and
                obj.cy - obj.height/2 >= region_top and obj.cy + obj.height/2 <= region_bottom):

                new_word.append(obj)
                frame.objects.pop(i)
            else:
                i += 1
        
        words_list.append(new_word)
    
    for word in words_list:
        if len(word) > 1:
            frame.words.append(word)
        else:
            frame.objects += word


def align_words(
    frame           : AnnotatedFrame,
    prefix_classes  : List[Set[int]],
    suffix_classes  : Set[int],
    y_tolerance     : float,
    x_tolerance     : float,
    mode            : str,
    create_groups   : bool = True):

    words_list = list()
    
    while True:
        new_word    : List[AnnotatedObject] = list()

        # find leftmost object that belongs to character classes
        if len(new_word) < len(prefix_classes):
            char_classes = prefix_classes[len(new_word)]
        else:
            char_classes = suffix_classes
        
        best_fit_index : int = -1
        for i, obj in enumerate(frame.objects):
            if obj.class_id not in char_classes:
                continue
            if best_fit_index < 0 or obj.cx < frame.objects[best_fit_index].cx:
                best_fit_index = i
        
        if best_fit_index < 0:
            break

        cy_sum      : float = 0
        height_sum  : float = 0

        # append adjacent characters
        while True:
            last_char : AnnotatedObject = frame.objects.pop(best_fit_index)
            new_word.append(last_char)
            cy_sum      += last_char.cy
            height_sum  += last_char.height

            if len(new_word) < len(prefix_classes):
                char_classes = prefix_classes[len(new_word)]
            else:
                char_classes = suffix_classes
            
            best_fit_index      : int = -1
            best_align_error    : float = x_tolerance
            word_mean_cy        : float = cy_sum / len(new_word)
            word_mean_height    : float = height_sum / len(new_word)

            for i, obj in enumerate(frame.objects):
                if obj.class_id not in char_classes:
                    continue

                if (abs((obj.cy - obj.height/2) - (word_mean_cy - word_mean_height/2)) > word_mean_height * y_tolerance or
                    abs((obj.cy + obj.height/2) - (word_mean_cy + word_mean_height/2)) > word_mean_height * y_tolerance):
                    continue

                align_error = abs((obj.cx - obj.width/2) - (last_char.cx + last_char.width/2)) / ((obj.width + last_char.width) * 0.5)
                if align_error < best_align_error:
                    best_fit_index = i
                    best_align_error = align_error
                
            if best_fit_index < 0:
                break
        
        # align characters
        if len(new_word) > 1:
            word_mean_cy        : float = cy_sum / len(new_word)
            word_mean_height    : float = height_sum / len(new_word)

            if mode == "adjacent":
                x_coords = list()
                x_coords.append(new_word[0].cx - new_word[0].width/2)
                for i in range(0, len(new_word)-1):
                    x_coords.append((
                        new_word[i].cx + new_word[i].width/2 +
                        new_word[i+1].cx - new_word[i+1].width/2
                    ) / 2)
                x_coords.append(new_word[-1].cx + new_word[-1].width/2)

                for i,obj in enumerate(new_word):
                    obj.cy      = word_mean_cy
                    obj.height  = word_mean_height
                    obj.cx      = (x_coords[i] + x_coords[i+1]) / 2
                    obj.width   = x_coords[i+1] - x_coords[i]

            elif mode == "uniform_spacing":
                width_sum   : float = 0
                cx_sum      : float = 0
                for obj in new_word:
                    width_sum   += obj.width
                    cx_sum      += obj.cx
                char_mean_width = width_sum / len(new_word)
                mean_cx         = cx_sum / len(new_word)

                sum_xi : float = 0
                sum_ii : float = 0

                for i,obj in enumerate(new_word):
                    i = i - (len(new_word)-1) / 2
                    sum_xi += (obj.cx - mean_cx) * i
                    sum_ii += i * i
                
                stride = sum_xi / sum_ii

                for i,obj in enumerate(new_word):
                    obj.cy      = word_mean_cy
                    obj.height  = word_mean_height
                    obj.cx      = mean_cx + (i - (len(new_word)-1) / 2) * stride
                    obj.width   = char_mean_width

            else:
                raise Exception("Unsupported alignment mode: '{}'".format(mode))
        
        words_list.append(new_word)
    
    for word in words_list:
        if create_groups and len(word) > 1:
            frame.words.append(word)
        else:
            frame.objects += word

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-dest", type = str)
    args = ap.parse_args()

    src_dir     : str = args.src
    dest_dir    : str = args.dest

    class_names = read_class_list(os.path.join(src_dir, "obj.names"))
    name_to_class_id : Dict[str, int] = dict()
    for i, name in enumerate(class_names):
        name_to_class_id[name] = i
    
    #
    number_char_classes     : Set[int] = set([name_to_class_id[n] for n in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]])
    card_partial_class      : int = name_to_class_id["CardPartial"]
    card_char_class         : int = name_to_class_id["Card"]
    blind_partial_class     : int = name_to_class_id["BlindPartial"]
    blind_char_class        : int = name_to_class_id["Blind"]
    rank_classes            : Set[int] = set([name_to_class_id[n] for n in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "J", "Q", "K", "A"]])
    suit_classes            : Set[int] = set([name_to_class_id[n] for n in ["Club", "Diamond", "Spade", "Heart"]])

    os.makedirs(os.path.join(dest_dir, "obj_train_data"), exist_ok = True)

    #
    for filename in os.listdir(os.path.join(src_dir, "obj_train_data")):
        _, ext = os.path.splitext(filename)
        if ext != ".txt":
            continue

        print(filename)
        frame = AnnotatedFrame()
        frame.objects = read_object_list(os.path.join(src_dir, "obj_train_data", filename))

        align_words(frame, [{card_partial_class }], {card_char_class  }, x_tolerance = 0.3, y_tolerance = 0.2, mode = "adjacent", create_groups = False)
        align_words(frame, [{blind_partial_class}], {blind_char_class }, x_tolerance = 0.3, y_tolerance = 0.2, mode = "adjacent", create_groups = False)
        align_words(frame, []                     , {card_char_class  }, x_tolerance = 0.3, y_tolerance = 0.2, mode = "uniform_spacing", create_groups = False)
        align_words(frame, []                     , {blind_char_class }, x_tolerance = 0.3, y_tolerance = 0.2, mode = "uniform_spacing", create_groups = False)

        group_card_suit_rank(frame, card_char_class, card_partial_class, rank_classes, suit_classes)
        align_words(frame, [], number_char_classes, x_tolerance = 0.3, y_tolerance = 0.3, mode = "adjacent")

        for word in frame.words:
            print("    ", end = "")
            for ch in word:
                print("{} ".format(class_names[ch.class_id]), end = "")
            print()
        
        write_object_list(os.path.join(dest_dir, "obj_train_data", filename), frame.to_list())

if __name__ == "__main__":
    main()
