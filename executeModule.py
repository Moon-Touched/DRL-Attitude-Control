from Basilisk.moduleTemplates import cModuleTemplate
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros


def run():
    """
    Controlling the simulation time
    """

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")

    # create the dynamics task and specify the integration update time
    dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", macros.sec2nano(1.)))

    # create modules
    mod1 = cModuleTemplate.cModuleTemplateConfig()
    mod1Wrap = scSim.setModelDataWrap(mod1)
    mod1Wrap.ModelTag = "cModule1"
    scSim.AddModelToTask("dynamicsTask", mod1Wrap, mod1)
    mod1.dummy = -10
    print(mod1.dummy)

    #  initialize Simulation:
    scSim.InitializeSimulation()
    print(mod1.dummy)

    # perform a single Update on all modules
    scSim.TotalSim.SingleStepProcesses()
    print(mod1.dummy)

    return


if __name__ == "__main__":
    run()
