import sys
import time

import numpy as np
import pandas as pd

from copy import deepcopy
from util.util import log_print


class Agent:

    def __init__(self, id, name, type, random_state):
        self.total_message_bits = 0
        self.id = id
        self.name = name
        self.type = type
        self.random_state = random_state
        self.message_stats = {
            "Seed sharing": {"count": 0, "bits": 0},
            "Legal clients confirmation": {"count": 0, "bits": 0},
            "Masked model upload": {"count": 0, "bits": 0},
            "Model aggregation and mask removal": {"count": 0, "bits": 0},
            "Online clients confirmation": {"count": 0, "bits": 0}
        }

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required " +
                             "for every agent.Agent", self.name)
            sys.exit()

        self.kernel = None
        self.currentTime = None
        self.log = []
        self.logEvent("AGENT_TYPE", type)

    ### Flow of required kernel listening methods:
    ### init -> start -> (entire simulation) -> end -> terminate

    def kernelInitializing(self, kernel):
        # Called by kernel one time when simulation first begins.
        # No other agents are guaranteed to exist at this time.

        # Kernel reference must be retained, as this is the only time the
        # agent can "see" it.
        self.kernel = kernel

        log_print("{} exists!", self.name)

    def kernelStarting(self, startTime):
        # Called by kernel one time _after_ simulationInitializing.
        # All other agents are guaranteed to exist at this time.
        # startTime is the earliest time for which the agent can
        # schedule a wakeup call (or could receive a message).

        # Base Agent schedules a wakeup call for the first available timestamp.
        # Subclass agents may override this behavior as needed.

        log_print("Agent {} ({}) requesting kernel wakeup at time {}",
                  self.id, self.name, self.kernel.fmtTime(startTime))

        self.setWakeup(startTime)

    def kernelStopping(self):
        # Called by kernel one time _before_ simulationTerminating.
        # All other agents are guaranteed to exist at this time.

        pass

    def kernelTerminating(self):
        # Called by kernel one time when simulation terminates.
        # No other agents are guaranteed to exist at this time.

        # If this agent has been maintaining a log, convert it to a Dataframe
        # and request that the Kernel write it to disk before terminating.
        if self.log:
            dfLog = pd.DataFrame(self.log)
            dfLog.set_index('EventTime', inplace=True)
            self.writeLog(dfLog)

    ### Methods for internal use by agents (e.g. bookkeeping).

    def logEvent(self, eventType, event='', appendSummaryLog=False):
        # Adds an event to this agent's log.  The deepcopy of the Event field,
        # often an object, ensures later state changes to the object will not
        # retroactively update the logged event.

        # We can make a single copy of the object (in case it is an arbitrary
        # class instance) for both potential log targets, because we don't
        # alter logs once recorded.
        e = deepcopy(event)
        self.log.append({'EventTime': self.currentTime, 'EventType': eventType,
                         'Event': e})

        if appendSummaryLog: self.kernel.appendSummaryLog(self.id, eventType, e)

    ### Methods required for communication from other agents.
    ### The kernel will _not_ call these methods on its own behalf,
    ### only to pass traffic from other agents..

    def receiveMessage(self, currentTime, msg):
        # Called each time a message destined for this agent reaches
        # the front of the kernel's priority queue.  currentTime is
        # the simulation time at which the kernel is delivering this
        # message -- the agent should treat this as "now".  msg is
        # an object guaranteed to inherit from the message.Message class.

        self.currentTime = currentTime

        log_print("At {}, agent {} ({}) received: {}",
                  self.kernel.fmtTime(currentTime), self.id, self.name, msg)

    def wakeup(self, currentTime):
        # Agents can request a wakeup call at a future simulation time using
        # Agent.setWakeup().  This is the method called when the wakeup time
        # arrives.

        self.currentTime = currentTime

        log_print("At {}, agent {} ({}) received wakeup.",
                  self.kernel.fmtTime(currentTime), self.id, self.name)

    ### Methods used to request services from the Kernel.  These should be used
    ### by all agents.  Kernel methods should _not_ be called directly!

    ### Presently the kernel expects agent IDs only, not agent references.
    ### It is possible this could change in the future.  Normal agents will
    ### not typically wish to request additional delay.
    def sendMessage(self, recipientID, msg, delay=0, tag="communication",msg_name=None):
        self.kernel.sendMessage(self.id, recipientID, msg, delay=delay, tag=tag)

    def print_message_stats(self):
        print("Message Statistics:")
        for message_type, stats in self.message_stats.items():
            print(f"- {message_type}: Count={stats['count']}, Total bits={stats['bits']}")

    def setWakeup(self, requestedTime):
        self.kernel.setWakeup(self.id, requestedTime)

    def getComputationDelay(self):
        return self.kernel.getAgentComputeDelay(sender=self.id)

    def setComputationDelay(self, requestedDelay):
        self.kernel.setAgentComputeDelay(sender=self.id, requestedDelay=requestedDelay)

    def delay(self, additionalDelay):
        self.kernel.delayAgent(sender=self.id, additionalDelay=additionalDelay)

    def writeLog(self, dfLog, filename=None):
        self.kernel.writeLog(self.id, dfLog, filename)

    def updateAgentState(self, state):
        """ Agents should use this method to replace their custom state in the dictionary
        the Kernel will return to the experimental config file at the end of the
        simulation.  This is intended to be write-only, and agents should not use
        it to store information for their own later use.
    """

        self.kernel.updateAgentState(self.id, state)

    ### Internal methods that should not be modified without a very good reason.

    def __lt__(self, other):
        # Required by Python3 for this object to be placed in a priority queue.

        return ("{}".format(self.id) <
                "{}".format(other.id))

