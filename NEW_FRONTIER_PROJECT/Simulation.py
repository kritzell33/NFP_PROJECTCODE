

import simpy
import mesa
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agents import ACTIONS, CrewAgent
from .events import EventDirector
from .social import RelationshipGraph


#Programmable by users

@dataclass
class Mission:
    designation: str
    length: int
    crewmodule: str
    onboard_science_experiments_per_mission: int
    eva_experiments_per_mission: int
    landing_zone: str
    space_station_workdays: int
    surface_experiments: int

class Hardware:
    name: str
    size: int
    location: int
    classification: int
    repair_knowledge_type: int

class Experiment:
    name: str
    knowledge_type: int
    deadline_day: int
    hardware: Hardware
    assigned_to: str

#General classes for simulation

#Experiments are a type of task.
class Task:
    kind: str
    effort_remaining: float
    created_0015: float
    deadline_0015: float | None = None   
    urgency = int
    completed_0015: float | None = None
    multi = bool
    claimed_by: str | None = None
    also_claimed_by: str | None = None     

         
class Simulation:
    def __init__(self, mission: Mission, agent: list[CrewAgent]):
        self.mission = mission
        self.agents = agent
        self.env = simpy.Enviroment
        self.model = mesa.Model


        grid = mesa.discrete_space.HexGrid((5, 4), torus=False)