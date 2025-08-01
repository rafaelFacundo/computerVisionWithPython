import json
import ijson
from decimal import Decimal


""" matchesJsonFile = open("./dataJson/Jleague_2024/sb_matches.json");

matchesData = json.load(matchesJsonFile); """

""" for key, value in matchesData[0].items():
    print(f"{key} : {value}") """

""" for match in matchesData:
    if match["match_week"] == 18:
        print(f"date {match["match_date"]}")
        print(f"match id {match["match_id"]}")
        print(f"matchweek {match["match_week"]}")
        print(
            f"{match["home_team.home_team_name"]} - {match["away_team.away_team_name"]}"
        )
        print("--------------")

matchesJsonFile.close(); """

matchesVideosInfosJsonFile = open("./matchesVideoInfos.json");

matchesVideosInfosJson = json.load(matchesVideosInfosJsonFile);

firstMatch = matchesVideosInfosJson[2]

firstMatchFirstHalfVideo = firstMatch["first_half_video_path"]
firstMatchId = firstMatch["match_id"]

eventsJsonFilePath = "./dataJson/Jleague_2024/sb_events.json"

matchEventsArray = [];

with open(eventsJsonFilePath, "r", encoding="utf-8") as f:
    print(1)
    events = ijson.items(f, "item")

    for event in events:
        if event.get("match_id") == firstMatchId:
            matchEventsArray.append(event)

outputFilePath = f"./jLeagueMatches/3925226/match_{firstMatchId}_events.json"

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Tipo {type(obj)} não é serializável")


with open(outputFilePath, "w",encoding="utf-8") as out_file:
    json.dump(matchEventsArray, out_file, indent=2, ensure_ascii=False, default=convert_decimal)


print(f"JSON salvo em: {outputFilePath}")