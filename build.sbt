import AssemblyKeys._

assemblySettings

name := "scalark"

version := "0.2"

organization := "org.scalark"

scalaVersion := "2.11.6"

resolvers += "Repo Spray IO" at "http://repo.spray.io"

libraryDependencies += "io.spray" %  "spray-json_2.11" % "1.3.2"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

libraryDependencies += "junit" % "junit" % "4.11" % "test"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.12"
