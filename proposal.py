from __future__ import annotations
from typing import Final
import json
import math
import statistics


# definitions for class identification
_0                  : Final[int]    = 0
_1                  : Final[int]    = 1
_2                  : Final[int]    = 2
_3                  : Final[int]    = 3
_4                  : Final[int]    = 4
_5                  : Final[int]    = 5
_6                  : Final[int]    = 6
_7                  : Final[int]    = 7
_8                  : Final[int]    = 8
_9                  : Final[int]    = 9
_A                  : Final[int]    = 10
_B                  : Final[int]    = 11
_C                  : Final[int]    = 12
_D                  : Final[int]    = 13
_E                  : Final[int]    = 14
_F                  : Final[int]    = 15
_G                  : Final[int]    = 16
_H                  : Final[int]    = 17
_I                  : Final[int]    = 18
_J                  : Final[int]    = 19
_K                  : Final[int]    = 20
_L                  : Final[int]    = 21
_M                  : Final[int]    = 22
_N                  : Final[int]    = 23
_O                  : Final[int]    = 24
_P                  : Final[int]    = 25
_Q                  : Final[int]    = 26
_R                  : Final[int]    = 27
_S                  : Final[int]    = 28
_T                  : Final[int]    = 29
_U                  : Final[int]    = 30
_V                  : Final[int]    = 31
_W                  : Final[int]    = 32
_X                  : Final[int]    = 33
_Y                  : Final[int]    = 34
_Z                  : Final[int]    = 35
_DOT                : Final[int]    = 36
_CLUB               : Final[int]    = 37
_DIAMOND            : Final[int]    = 38
_SPADE              : Final[int]    = 39
_HEART              : Final[int]    = 40
_CARD               : Final[int]    = 41
_SEAT               : Final[int]    = 42
_DOLLAR_SIGN        : Final[int]    = 43
_BLIND              : Final[int]    = 44
_POT                : Final[int]    = 45
_WAGE               : Final[int]    = 46
_DEALER             : Final[int]    = 47
_TIMER              : Final[int]    = 48
_TIMER_REMAINING    : Final[int]    = 49
_CARD_PARTIAL       : Final[int]    = 50
_BLIND_PARTIAL      : Final[int]    = 51


_NUMBERS        : Final[list[int]]  = [_0, _1, _2, _3, _4, _5, _6, _7, _8, _9]
_BANKROLLS      : Final[list[int]]  = _NUMBERS + [_DOT]
_CARD_KINDS     : Final[list[int]]  = [_CLUB, _DIAMOND, _SPADE, _HEART]
_CARD_KIND_NAMES : Final[list[str]] = {_CLUB: "Club", _DIAMOND: "Diamond", _SPADE: "Spade", _HEART: "Heart"}
_CARD_NAMES     : Final[list[int]]  = _NUMBERS + [_J, _Q, _K, _A]
_CARD_INFOS     : Final[list[int]]  = _CARD_NAMES + _CARD_KINDS
_CARDS          : Final[list[int]]  = [_CARD, _CARD_PARTIAL]
_BLINDS         : Final[list[int]]  = [_BLIND, _BLIND_PARTIAL]
_PLAYER_OBJECTS : Final[list[int]]  = _CARDS + _BLINDS + _BANKROLLS + _CARD_INFOS


# here are expanded boundary sizes
_EX_SZ_WAGE : Final[int] = 5
_CT_SZ_CARD : Final[int] = 10

class Boundary:
    x1 : float
    y1 : float
    x2 : float
    y2 : float

    def __init__(self, x1 : float = 0, y1 : float = 0, x2 : float = 0, y2 : float = 0):
        super().__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self) -> str:
        return "{:8.3f} {:8.3f}, {:3d} {:3d}".format(self.x1, self.y1, int(self.w()), int(self.h()))

    def w(self):
        return self.x2 - self.x1

    def h(self):
        return self.y2 - self.y1

    def mx(self):
        return (self.x1 + self.x2) / 2

    def my(self):
        return (self.y1 + self.y2) / 2

    def init_from(self, ob: Proposal):
        self.x1 = ob.x1
        self.y1 = ob.y1
        self.x2 = ob.x2
        self.y2 = ob.y2

    def expands(self, obj : Boundary):
        self.x1 = min(self.x1, obj.x1)
        self.y1 = min(self.y1, obj.y1)
        self.x2 = max(self.x2, obj.x2)
        self.y2 = max(self.y2, obj.y2)

    def contracts(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        self.x1 -= x1
        self.y1 -= y1
        self.x2 -= x2
        self.y2 -= y2

    def copy(self) -> Boundary:
        return Boundary(self.x1, self.y1, self.x2, self.y2)

    # check if two objects were intersected
    def doIntersect(self, obj : Boundary, be_sx : int = 0, be_sy : int = 0) -> bool:
        return (self.x1 - be_sx <= obj.x2 + be_sx and self.x2 + be_sx >= obj.x1 - be_sx and
                self.y1 - be_sy <= obj.y2 + be_sy and self.y2 + be_sy >= obj.y1 - be_sy)






class Card(Boundary):
    kind : int = None
    name : str = None

    def __init__(self, x1: float = 0, y1: float = 0, x2: float = 0, y2: float = 2):
        super().__init__(x1, y1, x2, y2)
        self.kind = None
        self.name = None

    def __str__(self) -> str:
        return "{} {}".format(
            _CARD_KIND_NAMES[self.kind] if self.kind != None else "",
            self.name if self.name != None else ""
        )

    @staticmethod
    def class_id_2_str(id):
        return "{}".format(id) if id in _NUMBERS else chr(id - _A + ord('A'))

    def init_from(self, ob: Proposal, obs: list[Proposal] = []):
        super().init_from(ob)
        self.contracts(y2 = _CT_SZ_CARD)
        names = []
        for ob in obs:
            if self.doIntersect(ob) == False:
                continue
            if ob.class_id in _CARD_NAMES:
                names.append(Card.class_id_2_str(ob.class_id))
            else:
                if ob.class_id in _CARD_KINDS:
                    self.kind = ob.class_id

        self.name = "".join(names)






# class `proposal` for object information
class Proposal(Boundary):
    class_id : int

    boundary : Boundary

    def __init__(self, class_id: int = None, x1: float = 0, y1: float = 0, x2: float = 0, y2: float = 0):
        super().__init__(x1, y1, x2, y2)
        self.class_id = class_id
        self.boundary = super().copy()

    def __str__(self) -> str:
        return "class_id {:3d} at {:8.3f} {:8.3f}, {:3d} {:3d}".format(self.class_id, self.x1, self.y1, int(self.w()), int(self.h()))


    # get the angle for the given seat
    @staticmethod
    def angle_of(ob : Proposal, c_pt : Point) -> float | None:
        return None if ob == None else math.atan2(ob.mx() - c_pt.x, c_pt.y - ob.my()) * 180 / math.pi


    # pick one from the given proposals and class id
    @staticmethod
    def pick_one_from(obs : list[Proposal], class_id : int) -> Proposal | None:
        for ob in obs:
            if ob.class_id == class_id:
                return ob
        return None


    # group by using intersected objects based on SOMETHING object
    @staticmethod
    def grouping_based_on_class_id(
            objects     : list[Proposal],
            class_id    : int               = _SEAT,
            group_ids   : list[int]         = [],
            be_sx       : int               = 0,
            be_sy       : int               = 0
        ) -> list[list[Proposal]]:

        groups : list[list[Proposal]] = []
        for i in range(len(objects))[::-1]:
            ob = objects[i]
            if ob.class_id == class_id:
                objects.pop(i)
                groups.append([ob])
                continue
            else:
                for group in groups:
                    tg = group[0]
                    if ob.class_id in group_ids and tg.boundary.doIntersect(ob, be_sx, be_sy):
                        objects.pop(i)
                        tg.boundary.expands(ob)
                        group.append(ob)
        for group in groups:
            group.sort(key = sort_by_x1)
        return groups


    # group by using intersected objects based on `SEAT` object
    @staticmethod
    def grouping_based_on_seat(objects : list[Proposal]) -> list[list[Proposal]]:
        return Proposal.grouping_based_on_class_id(objects, _SEAT, _PLAYER_OBJECTS)


    # group by using intersected objects based on `WAGE` object
    @staticmethod
    def grouping_based_on_wage(objects : list[Proposal]) -> list[list[Proposal]]:
        return Proposal.grouping_based_on_class_id(objects, _WAGE, _BANKROLLS, _EX_SZ_WAGE, _EX_SZ_WAGE)




class Base:
    id          : int
    boundary    : Boundary

    def __init__(self):
        super().__init__()




class Bankroll(Base):
    bankroll : float

    def __init__(self):
        super().__init__()
        self.bankroll = 0

    # parses bankroll from the given proposals
    def init_from(self, obs : list[Proposal], class_id : int, be_size : int = 0):
        boundary : Boundary = Proposal.pick_one_from(obs, class_id).copy()

        bankrolls : list[Proposal] = []
        for ob in list(filter(filter_4_bankrolls, obs)):
            if boundary.doIntersect(ob, be_size, 0):
                boundary.expands(ob)
                bankrolls.append(ob)

        bankrolls = "".join(list(map(map_id_2_str, bankrolls))).split(".")
        if len(bankrolls) == 1:
            self.bankroll = 0 if bankrolls[0] == "" else float("".join(bankrolls))
        else:
            decimals = bankrolls.pop()
            if len(decimals) >= 3:
                # this is for detect comma as dot
                self.bankroll = float("{}{}".format("".join(bankrolls), decimals))
            else:
                # ignores all thousand separators
                # includes only last dot
                self.bankroll = float("{}.{}".format("".join(bankrolls), decimals))




class Wage(Bankroll):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "wage {:11.2f}".format(self.bankroll)

    def init_from(self, obs : list[Proposal], pt : Point):
        super().init_from(obs, _WAGE, _EX_SZ_WAGE)
        ob: Proposal = Proposal.pick_one_from(obs, _WAGE)
        self.boundary = ob.boundary
        # self.id = int(Proposal.angle_of(ob, pt))




class Player(Bankroll):
    cards       : list[Card]    = None
    wage        : Wage          = None
    is_dealer   : bool
    is_playing  : bool

    def __init__(self):
        super().__init__()
        self.cards      = []
        self.is_dealer  = False
        self.is_playing = False

    def __str__(self) -> str:
        return "player {id:3d} | bankroll {bankroll:11.2f} | wage {wage:11.2f} | cards {cards}".format(
            id          = self.id,
            bankroll    = self.bankroll,
            wage        = self.wage.bankroll if self.wage != None else 0,
            cards      = ", ".join(str(card) for card in self.cards)
        )

    # init player from group that was made based on `Proposal` list
    def init_from(self, obs : list[Proposal], pt : Point):
        super().init_from(obs, _SEAT)
        ob: Proposal = Proposal.pick_one_from(obs, _SEAT)
        self.boundary = ob.boundary
        self.id = int(Proposal.angle_of(ob, pt))
        for ob in obs:
            if ob.class_id in _CARDS:
                card = Card()
                card.init_from(ob, obs)
                self.cards.append(card)
                self.is_playing = True
                continue
            if ob.class_id in _BLINDS:
                self.is_playing = True




class Point:
    x : float
    y : float

    def __init__(self, x : float = 0, y : float = 0):
        super().__init__()
        self.x = x
        self.y = y










# callback for sorting using `Proposal`.`x1`
def sort_by_x1(ob : Proposal) -> float:
    return ob.x1


# callback for filtering number and dot
def filter_4_bankrolls(ob : Proposal) -> bool:
    return ob.class_id in _BANKROLLS


# callback for getting middle point x of `Proposal`
def map_mx(ob : Proposal) -> float:
    return ob.mx()


# callback for getting middle point y of `Proposal`
def map_my(ob : Proposal) -> float:
    return ob.my()


# get the center point for `Proposal` list
def center_point(obs: list[Proposal]) -> Point:
    c_pt = Point()
    if len(obs) == 0:
        return c_pt
    c_pt.x = statistics.mean(list(map(map_mx, obs)))
    c_pt.y = statistics.mean(list(map(map_my, obs)))
    return c_pt


# callback for mapping class_id to str
def map_id_2_str(ob : Proposal) -> str:
    return "." if ob.class_id == _DOT else str(ob.class_id)


# callback for mapping to the 1st `Proposal` of the given list
def map_i0(obs : list[Proposal]) -> Proposal:
    return obs[0]


# callback for sorting models using their `id`
def sort_id(pl : Base) -> int:
    return pl.id

# callback for sorting models using their area
def sort_area(ob: Proposal) -> int:
    return int(ob.w() + ob.h())


def parse(objects : list[Proposal]):
    objects.sort(key = sort_area)
    # parse players
    groups = Proposal.grouping_based_on_seat(objects)
    c_pt = center_point(list(map(map_i0, groups)))
    players : list[Player] = []
    for group in groups:
        player = Player()
        player.init_from(group, c_pt)
        players.append(player)
    players.sort(key = sort_id)

    # parse wages
    groups = Proposal.grouping_based_on_wage(objects)
    for group in groups:
        wage = Wage()
        wage.init_from(group, c_pt)
        for player in players:
            if wage.boundary.doIntersect(player.boundary, _EX_SZ_WAGE, _EX_SZ_WAGE):
                player.wage = wage
                break

    for i in range(len(players)):
        players[i].id = i + 1
        print(players[i])

proposals : list[Proposal] = []

def sample_data(json_file):
    with open("./annotations/_/" + json_file, 'r') as fp:
        data = json.loads(fp.read().strip())
        for datum in data:
            proposals.append(Proposal(
                datum["class"],
                datum["location"]["x1"],
                datum["location"]["y1"],
                datum["location"]["x2"],
                datum["location"]["y2"]
            ))

sample_data("00001543.json")

parse(proposals)
