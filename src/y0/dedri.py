from y0.dsl import Element, Variable, Intervention, CounterfactualVariable   

__all__ = [
    'Interval',
    'Fluent'
]
class Interval(Element):
    """Timing and ordering queries.  
    
    These are queries that ask when things happen and for how long they happen.  
    We’ve already seen a couple of queries about the length of fluents and about the “during” relationship.  
    “During” is one of 13 Allen relations (really six and their inverses, plus equals), shown here: """
    def __init__(self, name, start=0, end=100):
        self.name=name
        assert(start < end)
        self.start= start
        self.end=end
    
    def to_text(self) -> str:
        """Output this variable in the internal string format."""
        return f"{self.name}[{self.start}:{self.end}]"
    
    def to_latex(self) -> str:
        """Output this variable in the LaTeX string format.

        :returns: The LaTeX representation of this variable.
        """
        # if it ends with a number, use that as a subscript
        return f"{self.name}_{{{self.start}-{self.end}}}"
        
    def __lt__(self, other: Interval) -> bool:
        """X precedes Y
        ---X----
                  -----Y------
        """
        return self.end < other.start
    def __gt__(self, other: Interval) -> bool:
        return self.start > other.end
    def meets(self, other: Interval) -> bool:
        """ X meets Y
        ----X---
                ----Y---
        """
        return self.end == other.start
    def is_met_by(self, other: Interval) -> bool:
        """ X is met by Y
        ----Y---
                ----X---
        """
        return self.start == other.end
    def overlaps_with(self, other: Interval) -> bool:
        """ X overlaps Y
        ----X---
             ----Y---
        """
        return (self.start < other.start) and  (self.end > other.start) and (self.end < other.end)
    def is_overlapped_by(self, other: Interval) -> bool:
        """ X is overlapped by Y
        ----Y---
             ----X---
        """
        return (self.start < other.start) and  (self.end > other.start) and (self.end < other.end)
    def starts(self, other: Interval) -> bool:
        """ X starts Y
        ----X---
        ----Y---------
        """
        return (self.start == other.start) and (self.end < other.end)
    def is_started_by(self, other: Interval) -> bool:
        """ X is started by Y
        ----Y---
        ----X---------
        """
        return (self.start == other.start) and (other.end < self.end)
    def during(self, other: Interval) -> bool:
        """ X during Y
          ---X---
        ----Y---------
        """
        return (self.start > other.start) and (self.end < other.end)
    def contains(self, other: Interval) -> bool:
        """ X contains Y
          ---Y---
        ----X---------
        """
        return (self.start < other.start) and (self.end > other.end)
    def finishes(self, other: Interval) -> bool:
        """ X finishes Y
               ---X---
        ----Y---------
        """
        return (self.start < other.start) and (self.end == other.end)
    def is_finished_by(self, other: Interval) -> bool:
        """ X is finished by Y
               ---Y---
        ----X---------
        """
        return (self.start > other.start) and (self.end == other.end)
    def __eq__(self, other: Interval) -> bool:
        """ X is equal to Y
        ---X---
        ---Y---
        """
        return (self.start == other.start) and (self.end == other.end)

class Fluent(Element):
    """Any pattern that persists over time is called a fluent.  
    
    For example, the pattern of interest might be that the size of the cohort of sick people is greater than 10% of the population.  
    If this persists over time, it is a fluent.   
    In logical approaches to reasoning about actions, fluents can be represented in first-order logic by predicates having an argument that depends on time. 
    For example, the condition "the box is on the table", if it can change over time, cannot be represented by 
    {\displaystyle \mathrm {On} (\mathrm {box} ,\mathrm {table} )}{\displaystyle \mathrm {On} (\mathrm {box} ,\mathrm {table} )}; 
    a third argument is necessary to the predicate {\displaystyle \mathrm {On} }{\displaystyle \mathrm {On} } to specify the time: 
    {\displaystyle \mathrm {On} (\mathrm {box} ,\mathrm {table} ,t)}{\displaystyle \mathrm {On} (\mathrm {box} ,\mathrm {table} ,t)} 
    means that the box is on the table at time {\displaystyle t}t. 
    This representation of fluents is modified in the situation calculus by using the sequence of the past actions in place of the current time.

    https://en.wikipedia.org/wiki/Fluent_(artificial_intelligence)
    """
    def __init__(self, name, interval: Interval):
        self.name == name
        self.interval == interval

    def to_text(self):
        """Output this fluent in the internal string format."""
        return f"{self.name}[{self.interval.to_latex()}]"
    
    def to_latex(self):
        """Output this fluent in the LaTeX string format.

        :returns: The LaTeX representation of this fluent.
        """
        return rf"\texttt{{{self.name}}}\left({self.interval.to_latex()}\right)"
    
    
