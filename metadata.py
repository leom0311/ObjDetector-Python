from typing import List, Tuple
import math

class ObjectDetectorMetadata:
    class_names         : List[str]

    num_mip_levels      : int
    first_level_scale   : int
    prior_object_size   : float     # area of prior rectangle, scaled
    max_aspect_ratio    : float
    num_aspect_levels   : int

    def calc_prior_box_sizes(self) -> List[Tuple[float, float]]:
        log_max_aspect_ratio = math.log(self.max_aspect_ratio)

        result = []
        for aspect_level in range(0, self.num_aspect_levels):
            aspect_ratio = math.exp(log_max_aspect_ratio * (aspect_level / (self.num_aspect_levels - 1) * 2 - 1))

            result.append((
                math.sqrt(self.prior_object_size * aspect_ratio), # width
                math.sqrt(self.prior_object_size / aspect_ratio)  # height
            ))
        return result
    
    # Converts box size [W, H] to [MipLevel, AspectLevel]
    def decode_box_size(self, width : float, height : float) -> Tuple[float, float]:
        level  = math.log(width * height / self.prior_object_size) / math.log(4) - 1
        aspect = (math.log(width / height) / math.log(self.max_aspect_ratio) + 1) * (self.num_aspect_levels - 1) / 2
        return level, aspect


OBJ_DETECTOR_MD_V1 = ObjectDetectorMetadata()
OBJ_DETECTOR_MD_V1.class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ".", "$",
    "Club", "Diamond", "Spade", "Heart",
    "Seat", "Card", "Blind", "Pot",
    "Wage", "Dealer", "CardPartial", "BlindPartial",
]
OBJ_DETECTOR_MD_V1.num_mip_levels       = 6
OBJ_DETECTOR_MD_V1.first_level_scale    = 2
OBJ_DETECTOR_MD_V1.prior_object_size    = 4.
OBJ_DETECTOR_MD_V1.max_aspect_ratio     = 4.
OBJ_DETECTOR_MD_V1.num_aspect_levels    = 7


if __name__ == "__main__":
    print(OBJ_DETECTOR_MD_V1.calc_prior_box_sizes())
