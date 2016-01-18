import sbt._
import sbt.Keys._

object BuildSettings {
  val buildName = "statnlpbook"
  val buildOrganization = "uk.ac.ucl.cs.mr"
  val buildScalaVersion = "2.11.7"


  val buildSettings = Defaults.defaultSettings ++ Seq(
    organization := buildOrganization,
    scalaVersion := buildScalaVersion,
    scalacOptions := Seq("-unchecked", "-deprecation", "-feature"), //, "-Yrangepos"?
    libraryDependencies ++= Seq(
      "org.scalanlp" %% "breeze" % "0.11.2",
      // native libraries are not included by default. add this if you want them (as of 0.7)
      // native libraries greatly improve performance, but increase jar sizes.
      // It also packages various blas implementations, which have licenses that may or may not
      // be compatible with the Apache License. No GPL code, as best I know.
      "org.scalanlp" %% "breeze-natives" % "0.11.2",
      "com.github.tototoshi" %% "scala-csv" % "1.3.0-SNAPSHOT",
      "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
    ),



  //shellPrompt := ShellPrompt.buildShellPrompt,
    fork in run := true, //use a fresh JVM for sbt run
    connectInput in run := true, //to use readLine after sbt run
    commands ++= Seq(vmargs),
    resolvers ++= Seq(
      "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    )
    //addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0-M1" cross CrossVersion.full)
  )

  val generalSettings =
    Seq(
      initialCommands := """
        import ml.wolfe.Wolfe._
        import ml.wolfe.macros.OptimizedOperators._
                         												 """
    )

  def vmargs = Command.args("vmargs", "<name>") {
    (state, args) =>
      val javaRunOptions = args.mkString(" ")
      println("Applying JVM arguments: " + javaRunOptions)
      Project.extract(state).append(javaOptions := Seq(javaRunOptions), state)
  }
}

object Build extends Build {

  import BuildSettings._

  lazy val statnlpbook = Project(
    id = "statnlpbook",
    base = file("."),
    settings = buildSettings
  ) dependsOn()
}