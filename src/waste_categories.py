"""
EcoVision Lite - Waste Classification Categories and Segregation Bin Guide.

This module maps raw dataset class names to user-friendly titles,
appropriate segregation bins, color codes, and eco-friendly disposal guidelines.
"""

# Segregation Bin Types:
# - Blue: Dry Recyclable (Paper, Cardboard, Clean Plastic, Dry Metals)
# - Green: Wet Organic / Compostable (Food waste, biological matter, leaves)
# - Red / Amber: Hazardous / E-Waste (Batteries, electronics, chemicals, sharps)
# - Yellow: Glass & Bottles (Dedicated glass recycling)
# - Purple / Brown: Textiles / Donations (Clothes, shoes, fabric)
# - Black / Grey: General Waste / Landfill (Non-recyclable contaminated items, trash)

BIN_DEFINITIONS = {
    "dry_recyclable": {
        "name": "Dry Recyclable Bin",
        "color": "#1E88E5",
        "icon": "♻️",
        "description": "Clean, dry materials that can be remanufactured into new products.",
        "badge_style": "background-color: #E3F2FD; color: #0D47A1; border: 1px solid #90CAF9;"
    },
    "wet_organic": {
        "name": "Organic / Compost Bin",
        "color": "#2E7D32",
        "icon": "🌱",
        "description": "Biodegradable organic waste suitable for aerobic composting or biogas generation.",
        "badge_style": "background-color: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7;"
    },
    "hazardous": {
        "name": "Hazardous & E-Waste Bin",
        "color": "#D32F2F",
        "icon": "⚠️",
        "description": "Special collection required. Contains toxic chemicals or heavy metals.",
        "badge_style": "background-color: #FFEBEE; color: #B71C1C; border: 1px solid #EF9A9A;"
    },
    "glass": {
        "name": "Glass Recycling Bin",
        "color": "#00897B",
        "icon": "🍾",
        "description": "100% endlessly recyclable glass containers and jars. Handle with care.",
        "badge_style": "background-color: #E0F2F1; color: #004D40; border: 1px solid #80CBC4;"
    },
    "textile": {
        "name": "Textile & Donation Bin",
        "color": "#8E24AA",
        "icon": "👕",
        "description": "Wearable garments or scrap fabrics for donation, re-purposing, or shredding.",
        "badge_style": "background-color: #F3E5F5; color: #4A148C; border: 1px solid #CE93D8;"
    },
    "general_waste": {
        "name": "General Landfill Bin",
        "color": "#546E7A",
        "icon": "🗑️",
        "description": "Non-recyclable or composite contaminated municipal solid waste.",
        "badge_style": "background-color: #ECEFF1; color: #263238; border: 1px solid #B0BEC5;"
    }
}

# Mapping dataset classes to rich metadata
# The dataset has classes: battery, biological, cardboard, clothes, glass (brown-glass, green-glass, white-glass),
# metal, paper, plastic, shoes, trash.
CATEGORY_DETAILS = {
    "battery": {
        "display_name": "Battery / Electronic Cell",
        "bin_key": "hazardous",
        "bin_recommendation": "Red Hazardous / E-Waste Bin",
        "instructions": [
            "Never dispose in regular household garbage or recycling bins.",
            "Cover positive and negative terminals with non-conductive clear tape.",
            "Drop off at a certified battery recycling depot, e-waste kiosk, or electronics retailer."
        ],
        "environmental_impact": "Batteries contain toxic heavy metals (lead, cadmium, lithium) that can leach into groundwater if sent to landfill.",
        "recycling_potential": "High. Lithium, cobalt, nickel, and steel can be recovered and reused."
    },
    "biological": {
        "display_name": "Biological / Organic Waste",
        "bin_key": "wet_organic",
        "bin_recommendation": "Green Compost / Wet Waste Bin",
        "instructions": [
            "Keep separate from plastics, foil, and packaging materials.",
            "Use for backyard composting or municipal aerobic digestion.",
            "Avoid adding excess plastic bags; use certified biodegradable liners if needed."
        ],
        "environmental_impact": "When buried in anaerobic landfills, organic matter generates methane, a potent greenhouse gas. Composting sequesters carbon.",
        "recycling_potential": "Excellent. Converts into nutrient-rich compost and soil conditioner."
    },
    "cardboard": {
        "display_name": "Cardboard Box / Packaging",
        "bin_key": "dry_recyclable",
        "bin_recommendation": "Blue Dry Recyclable Bin",
        "instructions": [
            "Flatten and break down boxes to maximize bin space.",
            "Remove excessive adhesive packing tape, plastic bubble wrap, and styrofoam.",
            "Keep dry! Greasy pizza cardboard should have unsoiled parts recycled, soiled parts composted."
        ],
        "environmental_impact": "Recycling one ton of cardboard saves approximately 9 cubic yards of landfill space and 17 trees.",
        "recycling_potential": "High. Cardboard fibers can be recycled 5 to 7 times."
    },
    "clothes": {
        "display_name": "Clothing / Fabric Garment",
        "bin_key": "textile",
        "bin_recommendation": "Purple Textile / Donation Drop-off",
        "instructions": [
            "If clean and wearable, donate to local charities, thrift stores, or clothing drives.",
            "If damaged or stained, deposit in specialized textile recycling boxes.",
            "Do not throw in general mixed recycling bins as fibers jam sorting machines."
        ],
        "environmental_impact": "Textile manufacturing consumes massive amounts of water and petroleum for synthetics. Extending garment life saves thousands of liters.",
        "recycling_potential": "Medium to High. Can be resold or shredded into industrial insulation and cleaning rags."
    },
    "glass": {
        "display_name": "Glass Container / Bottle",
        "bin_key": "glass",
        "bin_recommendation": "Yellow / Dedicated Glass Bin",
        "instructions": [
            "Empty all liquids and lightly rinse containers.",
            "Remove metal lids or plastic caps and place them in metal/plastic recycling.",
            "Do not mix with ceramics, light bulbs, window panes, or Pyrex."
        ],
        "environmental_impact": "Glass never decomposes in landfills, but it is 100% endlessly recyclable without loss in quality.",
        "recycling_potential": "100% Infinite. Remelted into new jars and bottles with 30% less energy than raw sand."
    },
    "brown-glass": {
        "display_name": "Brown / Amber Glass Container",
        "bin_key": "glass",
        "bin_recommendation": "Yellow / Amber Glass Bin",
        "instructions": [
            "Empty and rinse remaining liquid.",
            "Remove metal crown or screw cap.",
            "Keep separate by color if your local municipality has color-sorted glass banks."
        ],
        "environmental_impact": "Amber glass protects beverage contents from UV light and can be endlessly remelted into new amber bottles.",
        "recycling_potential": "100% Infinite recyclable."
    },
    "green-glass": {
        "display_name": "Green Glass Container",
        "bin_key": "glass",
        "bin_recommendation": "Yellow / Green Glass Bin",
        "instructions": [
            "Rinse residue thoroughly.",
            "Remove corks, plastic sleeves, and metallic collars.",
            "Recycle with colored container glass."
        ],
        "environmental_impact": "Cullet (crushed recycled glass) lowers melting temperatures, significantly reducing CO2 emissions in glass furnaces.",
        "recycling_potential": "100% Infinite recyclable."
    },
    "white-glass": {
        "display_name": "Clear / Flint Glass Container",
        "bin_key": "glass",
        "bin_recommendation": "Yellow / Clear Glass Bin",
        "instructions": [
            "Rinse clean of sauces or syrups.",
            "Metal twist lids can be placed in metal recycling.",
            "Deposit in clear glass recycling station."
        ],
        "environmental_impact": "High market demand for clear glass cullet to create new food jars.",
        "recycling_potential": "100% Infinite recyclable."
    },
    "metal": {
        "display_name": "Metal Can / Foil / Tin",
        "bin_key": "dry_recyclable",
        "bin_recommendation": "Blue Dry Recyclable Bin",
        "instructions": [
            "Rinse food and beverage cans clean.",
            "Aluminum foil can be balled up (larger balls are easier for optical sorters).",
            "Crush aluminum cans if permitted to save storage space."
        ],
        "environmental_impact": "Recycling aluminum uses 95% less energy than producing primary aluminum from bauxite ore.",
        "recycling_potential": "Infinite. Steel and aluminum can be melted down repeatedly without degradation."
    },
    "paper": {
        "display_name": "Paper / Documents / Newspaper",
        "bin_key": "dry_recyclable",
        "bin_recommendation": "Blue Dry Recyclable Bin",
        "instructions": [
            "Keep clean and dry; wet or contaminated paper is rejected.",
            "Remove metal binder clips or spiral wire bindings.",
            "Shredded paper should be bagged in a paper bag or checked against local rules."
        ],
        "environmental_impact": "Paper recycling saves trees, gallons of water, and electricity while reducing air pollutants.",
        "recycling_potential": "High. Paper fibers can be recycled 5 to 7 times into tissues and cartons."
    },
    "plastic": {
        "display_name": "Plastic Bottle / Container",
        "bin_key": "dry_recyclable",
        "bin_recommendation": "Blue Dry Recyclable Bin",
        "instructions": [
            "Empty all liquids and rinse food residue.",
            "Replace the cap on the bottle so it doesn't get lost in sorting.",
            "Check the resin code (PET #1 and HDPE #2 are most widely accepted)."
        ],
        "environmental_impact": "Plastics take 400+ years to break down and fragment into microplastics. Sorting correctly keeps them out of oceans.",
        "recycling_potential": "High for PET bottles and HDPE milk jugs; can become fleece, carpet, or new bottles."
    },
    "shoes": {
        "display_name": "Footwear / Shoes",
        "bin_key": "textile",
        "bin_recommendation": "Purple Textile / Footwear Drop-off",
        "instructions": [
            "Tie pairs together by the laces so they remain matched.",
            "Wearable shoes should be donated to community reuse centers.",
            "Worn out sneakers can often be recycled through athletic brand shoe recycling programs (e.g. playground turf)."
        ],
        "environmental_impact": "Complex multi-material construction (leather, rubber, EVA foam, adhesives) makes standard landfill disposal very persistent.",
        "recycling_potential": "Medium. Best handled via specialized brand take-back or charitable donation."
    },
    "trash": {
        "display_name": "General Non-Recyclable Waste",
        "bin_key": "general_waste",
        "bin_recommendation": "Black / Grey Landfill Bin",
        "instructions": [
            "Use for non-recyclable composite packaging, dirty wrappers, chip bags, broken ceramics, or hygiene products.",
            "Bag tightly to prevent litter dispersal during collection.",
            "Consider eco-friendly product alternatives to minimize single-use landfill trash."
        ],
        "environmental_impact": "Ends up in sanitary landfills or waste-to-energy incineration facilities.",
        "recycling_potential": "Low / Unrecyclable with current municipal infrastructure."
    }
}

def get_category_info(category_name: str) -> dict:
    """Retrieve full category metadata, bin info, and recommendations."""
    key = category_name.strip().lower()
    # Handle slight variations (e.g. 'glass' vs 'brown_glass', etc.)
    key_normalized = key.replace("_", "-")
    
    info = CATEGORY_DETAILS.get(key) or CATEGORY_DETAILS.get(key_normalized)
    if not info:
        # Fallback for unrecognized classes
        return {
            "display_name": category_name.capitalize(),
            "bin_key": "general_waste",
            "bin_recommendation": "General Waste Bin",
            "instructions": ["Check local recycling guidelines for this item."],
            "environmental_impact": "Proper segregation prevents landfill contamination.",
            "recycling_potential": "Check municipal facility instructions.",
            "bin_details": BIN_DEFINITIONS["general_waste"]
        }
    
    info_copy = dict(info)
    info_copy["bin_details"] = BIN_DEFINITIONS.get(info["bin_key"], BIN_DEFINITIONS["general_waste"])
    return info_copy
