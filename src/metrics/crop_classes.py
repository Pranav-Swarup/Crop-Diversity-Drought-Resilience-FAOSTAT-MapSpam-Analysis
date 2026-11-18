"""FAOSTAT crop item -> exposure class (one MapSPAM 2020 class, or a merged class where FAOSTAT cannot be split).

An exposure class is a tuple of MapSPAM 2020 crop codes (the 46 codes of the SPAM2020 v2r2 GeoTIFFs,
data/raw/mapspam/spam2020V2r2_global_physical_area.geotiff.zip, as in data/clean/crop_climate.parquet).
Merged classes:
  ("COFF", "RCOF")   "Coffee, green" (FAOSTAT does not split arabica and robusta)
  ("MILL", "PMIL")   "Millet" (FAOSTAT does not split small millets and pearl millet)

Source of the class definitions. The SPAM 2020 crop list that assigns FAO items to SPAM classes is in
Readme_SPAM2020V2r2.txt of the SPAM 2020 v2r2 release (Harvard Dataverse, https://doi.org/10.7910/DVN/SWPENT);
the file sits behind the Dataverse guestbook and could not be read when this module was written, so the
mapping below is built from the class names (dedicated classes) and the SPAM group names:
  OCER other cereals, ORTS other roots and tubers, OPUL other pulses, OOIL other oil crops, OFIB other fibre
  crops, TEMF temperate fruit, TROF tropical fruit, CITR citrus, VEGE vegetables, REST rest of crops.
Items whose class is a judgement call are in UNSURE (item_code -> reason).

The national drought-response and exposure-diversity analyses share this crosswalk.
"""
import pandas as pd

COFFEE = ("COFF", "RCOF")
MILLET = ("MILL", "PMIL")

# item_code: (FAOSTAT item name, exposure class)
ITEMS = {
    # cereals
    15: ("Wheat", ("WHEA",)),
    27: ("Rice", ("RICE",)),
    56: ("Maize (corn)", ("MAIZ",)),
    44: ("Barley", ("BARL",)),
    79: ("Millet", MILLET),
    83: ("Sorghum", ("SORG",)),
    71: ("Rye", ("OCER",)),
    75: ("Oats", ("OCER",)),
    89: ("Buckwheat", ("OCER",)),
    92: ("Quinoa", ("OCER",)),
    94: ("Fonio", ("OCER",)),
    97: ("Triticale", ("OCER",)),
    101: ("Canary seed", ("OCER",)),
    103: ("Mixed grain", ("OCER",)),
    108: ("Cereals n.e.c.", ("OCER",)),
    # roots and tubers
    116: ("Potatoes", ("POTA",)),
    122: ("Sweet potatoes", ("SWPO",)),
    125: ("Cassava, fresh", ("CASS",)),
    137: ("Yams", ("YAMS",)),
    135: ("Yautia", ("ORTS",)),
    136: ("Taro", ("ORTS",)),
    149: ("Edible roots and tubers with high starch or inulin content, n.e.c., fresh", ("ORTS",)),
    # sugar crops
    156: ("Sugar cane", ("SUGC",)),
    157: ("Sugar beet", ("SUGB",)),
    161: ("Other sugar crops n.e.c.", ("REST",)),
    # pulses
    176: ("Beans, dry", ("BEAN",)),
    191: ("Chick peas, dry", ("CHIC",)),
    195: ("Cow peas, dry", ("COWP",)),
    197: ("Pigeon peas, dry", ("PIGE",)),
    201: ("Lentils, dry", ("LENT",)),
    181: ("Broad beans and horse beans, dry", ("OPUL",)),
    187: ("Peas, dry", ("OPUL",)),
    203: ("Bambara beans, dry", ("OPUL",)),
    205: ("Vetches", ("OPUL",)),
    210: ("Lupins", ("OPUL",)),
    211: ("Other pulses n.e.c.", ("OPUL",)),
    # tree nuts
    217: ("Cashew nuts, in shell", ("REST",)),
    220: ("Chestnuts, in shell", ("REST",)),
    221: ("Almonds, in shell", ("REST",)),
    222: ("Walnuts, in shell", ("REST",)),
    223: ("Pistachios, in shell", ("REST",)),
    224: ("Kola nuts", ("REST",)),
    225: ("Hazelnuts, in shell", ("REST",)),
    226: ("Areca nuts", ("REST",)),
    234: ("Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.", ("REST",)),
    # oil crops
    236: ("Soya beans", ("SOYB",)),
    242: ("Groundnuts, excluding shelled", ("GROU",)),
    249: ("Coconuts, in shell", ("CNUT",)),
    254: ("Oil palm fruit", ("OILP",)),
    267: ("Sunflower seed", ("SUNF",)),
    270: ("Rape or colza seed", ("RAPE",)),
    289: ("Sesame seed", ("SESA",)),
    260: ("Olives", ("OOIL",)),
    263: ("Karite nuts (sheanuts)", ("OOIL",)),
    265: ("Castor oil seeds", ("OOIL",)),
    275: ("Tung nuts", ("OOIL",)),
    277: ("Jojoba seeds", ("OOIL",)),
    280: ("Safflower seed", ("OOIL",)),
    292: ("Mustard seed", ("OOIL",)),
    296: ("Poppy seed", ("OOIL",)),
    299: ("Melonseed", ("OOIL",)),
    305: ("Tallowtree seeds", ("OOIL",)),
    310: ("Kapok fruit", ("OOIL",)),
    333: ("Linseed", ("OOIL",)),
    336: ("Hempseed", ("OOIL",)),
    339: ("Other oil seeds, n.e.c.", ("OOIL",)),
    # fibre crops
    328: ("Seed cotton, unginned", ("COTT",)),
    771: ("Flax, raw or retted", ("OFIB",)),
    777: ("True hemp, raw or retted", ("OFIB",)),
    780: ("Jute, raw or retted", ("OFIB",)),
    782: ("Kenaf, and other textile bast fibres, raw or retted", ("OFIB",)),
    788: ("Ramie, raw or retted", ("OFIB",)),
    789: ("Sisal, raw", ("OFIB",)),
    800: ("Agave fibres, raw, n.e.c.", ("OFIB",)),
    809: ("Abaca, manila hemp, raw", ("OFIB",)),
    821: ("Other fibre crops, raw, n.e.c.", ("OFIB",)),
    # vegetables (FAOSTAT vegetables group, melons included)
    358: ("Cabbages", ("VEGE",)),
    366: ("Artichokes", ("VEGE",)),
    367: ("Asparagus", ("VEGE",)),
    372: ("Lettuce and chicory", ("VEGE",)),
    373: ("Spinach", ("VEGE",)),
    388: ("Tomatoes", ("TOMA",)),
    393: ("Cauliflowers and broccoli", ("VEGE",)),
    394: ("Pumpkins, squash and gourds", ("VEGE",)),
    397: ("Cucumbers and gherkins", ("VEGE",)),
    399: ("Eggplants (aubergines)", ("VEGE",)),
    401: ("Chillies and peppers, green (Capsicum spp. and Pimenta spp.)", ("VEGE",)),
    402: ("Onions and shallots, green", ("ONIO",)),
    403: ("Onions and shallots, dry (excluding dehydrated)", ("ONIO",)),
    406: ("Green garlic", ("VEGE",)),
    407: ("Leeks and other alliaceous vegetables", ("VEGE",)),
    414: ("Other beans, green", ("VEGE",)),
    417: ("Peas, green", ("VEGE",)),
    420: ("Broad beans and horse beans, green", ("VEGE",)),
    423: ("String beans", ("VEGE",)),
    426: ("Carrots and turnips", ("VEGE",)),
    430: ("Okra", ("VEGE",)),
    446: ("Green corn (maize)", ("VEGE",)),
    459: ("Chicory roots", ("REST",)),
    461: ("Locust beans (carobs)", ("REST",)),
    463: ("Other vegetables, fresh n.e.c.", ("VEGE",)),
    567: ("Watermelons", ("VEGE",)),
    568: ("Cantaloupes and other melons", ("VEGE",)),
    # bananas
    486: ("Bananas", ("BANA",)),
    489: ("Plantains and cooking bananas", ("PLNT",)),
    # citrus
    490: ("Oranges", ("CITR",)),
    495: ("Tangerines, mandarins, clementines", ("CITR",)),
    497: ("Lemons and limes", ("CITR",)),
    507: ("Pomelos and grapefruits", ("CITR",)),
    512: ("Other citrus fruit, n.e.c.", ("CITR",)),
    # temperate fruit
    515: ("Apples", ("TEMF",)),
    521: ("Pears", ("TEMF",)),
    523: ("Quinces", ("TEMF",)),
    526: ("Apricots", ("TEMF",)),
    530: ("Sour cherries", ("TEMF",)),
    531: ("Cherries", ("TEMF",)),
    534: ("Peaches and nectarines", ("TEMF",)),
    536: ("Plums and sloes", ("TEMF",)),
    541: ("Other stone fruits", ("TEMF",)),
    542: ("Other pome fruits", ("TEMF",)),
    544: ("Strawberries", ("TEMF",)),
    547: ("Raspberries", ("TEMF",)),
    549: ("Gooseberries", ("TEMF",)),
    550: ("Currants", ("TEMF",)),
    552: ("Blueberries", ("TEMF",)),
    554: ("Cranberries", ("TEMF",)),
    558: ("Other berries and fruits of the genus vaccinium n.e.c.", ("TEMF",)),
    560: ("Grapes", ("TEMF",)),
    569: ("Figs", ("TEMF",)),
    587: ("Persimmons", ("TEMF",)),
    592: ("Kiwi fruit", ("TEMF",)),
    # tropical fruit
    571: ("Mangoes, guavas and mangosteens", ("TROF",)),
    572: ("Avocados", ("TROF",)),
    574: ("Pineapples", ("TROF",)),
    577: ("Dates", ("TROF",)),
    591: ("Cashewapple", ("TROF",)),
    600: ("Papayas", ("TROF",)),
    603: ("Other tropical fruits, n.e.c.", ("TROF",)),
    619: ("Other fruits, n.e.c.", ("TROF",)),
    # stimulants
    656: ("Coffee, green", COFFEE),
    661: ("Cocoa beans", ("COCO",)),
    667: ("Tea leaves", ("TEAS",)),
    671: ("Maté leaves", ("REST",)),
    826: ("Unmanufactured tobacco", ("TOBA",)),
    # spices, hops, aromatic and other crops
    677: ("Hop cones", ("REST",)),
    687: ("Pepper (Piper spp.), raw", ("REST",)),
    689: ("Chillies and peppers, dry (Capsicum spp., Pimenta spp.), raw", ("REST",)),
    692: ("Vanilla, raw", ("REST",)),
    693: ("Cinnamon and cinnamon-tree flowers, raw", ("REST",)),
    698: ("Cloves (whole stems), raw", ("REST",)),
    702: ("Nutmeg, mace, cardamoms, raw", ("REST",)),
    711: ("Anise, badian, coriander, cumin, caraway, fennel and juniper berries, raw", ("REST",)),
    720: ("Ginger, raw", ("REST",)),
    723: ("Other stimulant, spice and aromatic crops, n.e.c.", ("REST",)),
    748: ("Peppermint, spearmint", ("REST",)),
    754: ("Pyrethrum, dried flowers", ("REST",)),
    836: ("Natural rubber in primary forms", ("RUBB",)),
}

# item_code -> why the class is uncertain (reported, not used in computation).
UNSURE = {
    260: "Olives: OOIL (FAOSTAT oil crop) vs REST",
    263: "Karite nuts: OOIL (FAOSTAT oil crop) vs REST (tree nut)",
    226: "Areca nuts: REST (nut) vs TROF",
    402: "Onions and shallots, green: ONIO vs VEGE",
    406: "Green garlic: VEGE vs ONIO",
    459: "Chicory roots: REST vs VEGE/ORTS",
    461: "Locust beans (carobs): REST vs TEMF",
    569: "Figs: TEMF vs TROF",
    587: "Persimmons: TEMF vs TROF",
    591: "Cashewapple: TROF vs REST (with cashew nuts)",
    619: "Other fruits, n.e.c.: TROF vs TEMF",
    689: "Chillies and peppers, dry: REST (spice) vs VEGE",
    161: "Other sugar crops n.e.c.: REST vs SUGC",
    567: "Watermelons: VEGE vs TROF",
    568: "Cantaloupes and other melons: VEGE vs TROF",
    446: "Green corn (maize): VEGE vs MAIZ",
}

SPAM_CODES = ("BANA BARL BEAN CASS CHIC CITR CNUT COCO COFF COTT COWP GROU LENT MAIZ MILL OCER OFIB OILP ONIO OOIL "
              "OPUL ORTS PIGE PLNT PMIL POTA RAPE RCOF REST RICE RUBB SESA SORG SOYB SUGB SUGC SUNF SWPO TEAS TEMF "
              "TOBA TOMA TROF VEGE WHEA YAMS").split()


def class_label(cls):
    """("COFF", "RCOF") -> "COFF+RCOF"."""
    return "+".join(cls)


def mapping_table():
    """DataFrame item_code, item, exposure_class (label), members (tuple), unsure (bool)."""
    rows = [{"item_code": k, "item": n, "exposure_class": class_label(c), "members": c, "unsure": k in UNSURE}
            for k, (n, c) in ITEMS.items()]
    return pd.DataFrame(rows).sort_values("item_code").reset_index(drop=True)


def check(crops, crop_climate_codes):
    """Assert every item_code in crops is mapped, every mapped code is a SPAM 2020 class present in
    crop_climate, and the mapped names match the FAOSTAT names (ignoring encoding of non-ASCII characters)."""
    codes = set(crops["item_code"].unique())
    missing = sorted(codes - set(ITEMS))
    assert not missing, f"FAOSTAT items without an exposure class: {missing}"
    members = {m for _, c in ITEMS.values() for m in c}
    assert members <= set(SPAM_CODES), f"codes that are not SPAM 2020 classes: {members - set(SPAM_CODES)}"
    assert members <= set(crop_climate_codes), f"codes absent from crop_climate: {members - set(crop_climate_codes)}"
    for code, name in crops.drop_duplicates("item_code").set_index("item_code")["item"].items():
        mine = ITEMS[code][0]
        same = mine.encode("ascii", "ignore") == str(name).encode("ascii", "ignore")
        assert same, f"item_code {code}: mapped name {mine!r} vs FAOSTAT {name!r}"
    return True
